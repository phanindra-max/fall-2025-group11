"""
Teacher-Student framework with EMA teacher and standard SSL losses
Matches the interfaces expected by the trainer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from tqdm import tqdm
import copy


class EMATeacherLegacy:
    """
    Maintains an exponential moving average (EMA) copy of the student model.
    """

    def __init__(self, student_model: nn.Module, ema_decay: float = 0.999, device: torch.device = torch.device('cpu')):
        self.device = device
        self.ema_decay = ema_decay
        # Create a deep copy of the student as the teacher (works for custom constructors)
        self.teacher_model = copy.deepcopy(student_model)
        self.teacher_model.to(self.device)
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, student_model: nn.Module):
        """EMA update: θ_T = β θ_T + (1-β) θ_S"""
        ema = self.ema_decay
        for ema_param, student_param in zip(self.teacher_model.parameters(), student_model.parameters()):
            ema_param.data.mul_(ema).add_(student_param.data, alpha=(1.0 - ema))


def _clone_model_by_state_dict(model: nn.Module) -> nn.Module:
    clone = type(model)(*[])
    clone.load_state_dict(model.state_dict())
    return clone


class TeacherStudentFramework:
    """
    Wraps student and EMA teacher models and exposes required utilities.
    """

    def __init__(self, student_model: nn.Module, ema_decay: float = 0.999, device: torch.device = torch.device('cpu')):
        self.device = device
        self.student_model = student_model.to(device)
        self.teacher = EMATeacher(self.student_model, ema_decay)
        self.ce_loss = nn.CrossEntropyLoss()

    @torch.no_grad()
    def forward_teacher(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        self.teacher.teacher_model.eval()
        if return_logits:
            return self.teacher.teacher_model.get_logits(x)
        return self.teacher.teacher_model.get_probabilities(x)

    def compute_supervised_loss(self, x_l: torch.Tensor, y_l: torch.Tensor) -> torch.Tensor:
        logits = self.student_model.get_logits(x_l)
        return self.ce_loss(logits, y_l)

    def compute_consistency_loss(self, x_u: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            teacher_probs = self.forward_teacher(x_u, return_logits=False)
        student_logits = self.student_model.get_logits(x_u)
        student_log_probs = F.log_softmax(student_logits, dim=1)
        # l_KL(P_s, P_T) with teacher as target distribution
        cons = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        return cons

    @torch.no_grad()
    def evaluate(self, test_loader) -> Dict[str, float]:
        self.student_model.eval()
        self.teacher.teacher_model.eval()

        student_correct = 0
        teacher_correct = 0
        total = 0

        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch[0], batch[1]
            else:
                images, labels = batch

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.no_grad():
                s_logits = self.student_model.get_logits(images)
                t_logits = self.teacher.teacher_model.get_logits(images)

            student_pred = s_logits.argmax(dim=1)
            teacher_pred = t_logits.argmax(dim=1)
            student_correct += (student_pred == labels).sum().item()
            teacher_correct += (teacher_pred == labels).sum().item()
            total += labels.size(0)

        return {
            'student_accuracy': 100.0 * student_correct / max(1, total),
            'teacher_accuracy': 100.0 * teacher_correct / max(1, total)
        }

    @torch.no_grad()
    def update_teacher(self):
        self.teacher.update(self.student_model)


def warmup_teacher_student(
    framework,
    labeled_loader,
    unlabeled_loader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    lambda_sup: float = 0.1,
    lambda_cons: float = 0.1,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
):
    """
    Warmup phase: standard teacher-student training with supervised + consistency loss.
    """
    device = next(framework.student_model.parameters()).device
    use_amp = scaler is not None and device.type == 'cuda'

    framework.student_model.train()

    for epoch in range(num_epochs):
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        progress = tqdm(total=min(len(labeled_loader), len(unlabeled_loader)), desc=f"Warmup {epoch+1}/{num_epochs}")

        while True:
            try:
                batch_l = next(labeled_iter)
                batch_u = next(unlabeled_iter)
            except StopIteration:
                break

            # Robustly unpack batches that may be (data, label) or (data, label, idx)
            if isinstance(batch_l, (list, tuple)):
                if len(batch_l) >= 2:
                    x_l, y_l = batch_l[0], batch_l[1]
                else:
                    x_l, y_l = batch_l[0], None
            else:
                x_l, y_l = batch_l, None

            if isinstance(batch_u, (list, tuple)):
                x_u = batch_u[0]
            else:
                x_u = batch_u

            x_l = x_l.to(device, non_blocking=True)
            if y_l is not None:
                y_l = y_l.to(device, non_blocking=True)
            x_u = x_u.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    sup = framework.compute_supervised_loss(x_l, y_l)
                    cons = framework.compute_consistency_loss(x_u)
                    loss = lambda_sup * sup + lambda_cons * cons
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sup = framework.compute_supervised_loss(x_l, y_l)
                cons = framework.compute_consistency_loss(x_u)
                loss = lambda_sup * sup + lambda_cons * cons
                loss.backward()
                optimizer.step()

            framework.update_teacher()
            progress.update(1)

        progress.close()

"""
Teacher-Student Framework for RLGSSL
Implements EMA-based teacher model for stable pseudo-label generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import copy


class EMATeacher:
    """
    Exponential Moving Average Teacher Model
    Maintains a stable version of the student model for pseudo-label generation
    """
    
    def __init__(self, student_model: nn.Module, ema_decay: float = 0.999):
        """
        Initialize EMA teacher
        
        Args:
            student_model: The student model to track
            ema_decay: EMA decay rate (β in the paper)
        """
        self.ema_decay = ema_decay
        self.student_model = student_model
        
        # Create teacher model as a copy of student
        self.teacher_model = copy.deepcopy(student_model)
        
        # Set teacher to eval mode and disable gradients
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def update(self):
        """
        Update teacher model parameters using EMA
        θ_T = β * θ_T + (1 - β) * θ_S
        """
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_model.parameters(), 
                self.student_model.parameters()
            ):
                teacher_param.data.mul_(self.ema_decay).add_(
                    student_param.data, alpha=1 - self.ema_decay
                )
    
    def __call__(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """Forward pass through teacher model"""
        self.teacher_model.eval()
        with torch.no_grad():
            return self.teacher_model(x, return_logits=return_logits)
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Get probabilities from teacher model"""
        return self(x, return_logits=False)
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get logits from teacher model"""
        return self(x, return_logits=True)


class TeacherStudentFramework:
    """
    Complete Teacher-Student Framework for RLGSSL
    Manages both student and teacher models with various loss components
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        ema_decay: float = 0.999,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize teacher-student framework
        
        Args:
            student_model: The student model to train
            ema_decay: EMA decay rate for teacher
            device: Device to run models on
        """
        self.student_model = student_model.to(device)
        self.teacher = EMATeacher(student_model, ema_decay)
        self.device = device
        
        # Move teacher to device
        self.teacher.teacher_model = self.teacher.teacher_model.to(device)
    
    def forward_student(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """Forward pass through student model"""
        return self.student_model(x, return_logits=return_logits)
    
    def forward_teacher(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """Forward pass through teacher model"""
        return self.teacher(x, return_logits=return_logits)
    
    def update_teacher(self):
        """Update teacher model with EMA"""
        self.teacher.update()
    
    def compute_supervised_loss(
        self, 
        labeled_data: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised loss on labeled data
        L_sup = E[(x^l, y^l) ∈ D^l] [l_CE(P_θs(x^l), y^l)]
        
        Args:
            labeled_data: Labeled input data
            labels: True labels
        
        Returns:
            Cross-entropy loss
        """
        student_logits = self.forward_student(labeled_data, return_logits=True)
        return F.cross_entropy(student_logits, labels)
    
    def compute_consistency_loss(
        self, 
        unlabeled_data: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute consistency loss between teacher and student predictions
        L_cons = E[x^u ∈ D^u] [l_KL(P_θs(x^u), P_θT(x^u))]
        
        Args:
            unlabeled_data: Unlabeled input data
            temperature: Temperature for softmax (for sharpening)
        
        Returns:
            KL divergence loss
        """
        # Get teacher predictions (stable, no gradients)
        teacher_probs = self.forward_teacher(unlabeled_data, return_logits=False)
        
        # Get student predictions (with gradients)
        student_logits = self.forward_student(unlabeled_data, return_logits=True)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        
        # KL divergence: KL(teacher || student)
        consistency_loss = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction='batchmean'
        )
        
        return consistency_loss
    
    def generate_pseudo_labels(
        self, 
        unlabeled_data: torch.Tensor,
        confidence_threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate pseudo-labels using teacher model
        
        Args:
            unlabeled_data: Unlabeled input data
            confidence_threshold: Optional confidence threshold for filtering
        
        Returns:
            Tuple of (pseudo_labels, confidence_mask)
        """
        teacher_probs = self.forward_teacher(unlabeled_data, return_logits=False)
        
        if confidence_threshold is not None:
            # Apply confidence threshold
            max_probs, pseudo_labels = torch.max(teacher_probs, dim=1)
            confidence_mask = max_probs > confidence_threshold
            return teacher_probs, confidence_mask
        else:
            # Return all pseudo-labels (soft labels)
            return teacher_probs, torch.ones(teacher_probs.size(0), dtype=torch.bool, device=self.device)
    
    def train_step(
        self,
        labeled_data: torch.Tensor,
        labels: torch.Tensor,
        unlabeled_data: torch.Tensor,
        lambda_sup: float = 1.0,
        lambda_cons: float = 1.0
    ) -> Dict[str, float]:
        """
        Perform a training step with supervised and consistency losses
        (This is the base SSL training, before adding RL components)
        
        Args:
            labeled_data: Labeled input data
            labels: True labels
            unlabeled_data: Unlabeled input data
            lambda_sup: Weight for supervised loss
            lambda_cons: Weight for consistency loss
        
        Returns:
            Dictionary of losses
        """
        # Compute losses
        sup_loss = self.compute_supervised_loss(labeled_data, labels)
        cons_loss = self.compute_consistency_loss(unlabeled_data)
        
        # Combined loss
        total_loss = lambda_sup * sup_loss + lambda_cons * cons_loss
        
        return {
            'supervised_loss': sup_loss.item(),
            'consistency_loss': cons_loss.item(),
            'total_loss': total_loss.item(),
            'combined_loss_tensor': total_loss  # Return tensor for backprop
        }
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """
        Evaluate both student and teacher models (robust to 2- or 3-tuple batches)
        """
        self.student_model.eval()
        
        student_correct = 0
        teacher_correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 2:
                        data, targets = batch[0], batch[1]
                    else:
                        continue
                else:
                    # Unsupported format
                    continue
                
                data = data.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Student predictions
                student_logits = self.forward_student(data, return_logits=True)
                student_pred = student_logits.argmax(dim=1)
                student_correct += (student_pred == targets).sum().item()
                
                # Teacher predictions
                teacher_logits = self.forward_teacher(data, return_logits=True)
                teacher_pred = teacher_logits.argmax(dim=1)
                teacher_correct += (teacher_pred == targets).sum().item()
                
                total += targets.size(0)
        
        student_accuracy = 100.0 * student_correct / max(1, total)
        teacher_accuracy = 100.0 * teacher_correct / max(1, total)
        
        return {
            'student_accuracy': student_accuracy,
            'teacher_accuracy': teacher_accuracy,
            'student_error': 100.0 - student_accuracy,
            'teacher_error': 100.0 - teacher_accuracy
        }


# Utility functions for different SSL training strategies
def warmup_teacher_student(
    framework: 'TeacherStudentFramework',
    labeled_loader,
    unlabeled_loader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 50,
    lambda_sup: float = 1.0,
    lambda_cons: float = 1.0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
):
    """
    Warmup training with optional AMP and robust batch unpacking.
    """
    device = framework.device if hasattr(framework, 'device') else next(framework.student_model.parameters()).device
    use_amp = scaler is not None and device.type == 'cuda'
    framework.student_model.train()
    
    for epoch in range(num_epochs):
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        while True:
            try:
                batch_l = next(labeled_iter)
                batch_u = next(unlabeled_iter)
            except StopIteration:
                break
            
            # Unpack batches (support 2- or 3-tuples)
            if isinstance(batch_l, (list, tuple)) and len(batch_l) >= 2:
                x_l, y_l = batch_l[0], batch_l[1]
            else:
                continue
            x_u = batch_u[0] if isinstance(batch_u, (list, tuple)) else batch_u
            
            x_l = x_l.to(device, non_blocking=True)
            y_l = y_l.to(device, non_blocking=True)
            x_u = x_u.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    sup = framework.compute_supervised_loss(x_l, y_l)
                    cons = framework.compute_consistency_loss(x_u)
                    loss = lambda_sup * sup + lambda_cons * cons
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sup = framework.compute_supervised_loss(x_l, y_l)
                cons = framework.compute_consistency_loss(x_u)
                loss = lambda_sup * sup + lambda_cons * cons
                loss.backward()
                optimizer.step()
            
            framework.update_teacher()


if __name__ == "__main__":
    # Test the teacher-student framework
    from ..models.networks import create_model
    
    # Create a simple test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model('cnn13', num_classes=10)
    
    framework = TeacherStudentFramework(model, ema_decay=0.999, device=device)
    
    # Test forward passes
    test_data = torch.randn(4, 3, 32, 32).to(device)
    test_labels = torch.randint(0, 10, (4,)).to(device)
    
    print("Testing teacher-student framework...")
    
    # Test supervised loss
    sup_loss = framework.compute_supervised_loss(test_data, test_labels)
    print(f"Supervised loss: {sup_loss.item():.4f}")
    
    # Test consistency loss
    cons_loss = framework.compute_consistency_loss(test_data)
    print(f"Consistency loss: {cons_loss.item():.4f}")
    
    # Test pseudo-label generation
    pseudo_labels, mask = framework.generate_pseudo_labels(test_data)
    print(f"Pseudo-labels shape: {pseudo_labels.shape}")
    print(f"Confidence mask: {mask.sum().item()}/{len(mask)} samples")
    
    # Test teacher update
    print("Testing teacher update...")
    old_teacher_param = list(framework.teacher.teacher_model.parameters())[0].clone()
    framework.update_teacher()
    new_teacher_param = list(framework.teacher.teacher_model.parameters())[0]
    
    param_change = torch.norm(new_teacher_param - old_teacher_param).item()
    print(f"Teacher parameter change: {param_change:.6f}")
    
    print("Teacher-student framework test completed!")