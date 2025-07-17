# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
import os
import torch
import hydra
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import LaunchLogger, PythonLogger, initialize_loggers
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint

from physicsnemo_solver import PhysicsNeMoPINNSolver


def create_trainer(cfg: DictConfig, solver: PhysicsNeMoPINNSolver):
    """Create PhysicsNeMo trainer with distributed support"""
    
    # Get boundary and interior data
    boundary_data = solver.dataset.get_boundary_data()
    interior_data = solver.dataset.get_interior_data()
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        solver.model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay
    )
    
    # Setup scheduler if specified
    scheduler = None
    if cfg.scheduler.enabled:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=cfg.scheduler.gamma
        )
    
    return optimizer, scheduler, boundary_data, interior_data


def training_step(
    solver: PhysicsNeMoPINNSolver,
    boundary_data: dict,
    interior_data: dict,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: DictConfig
):
    """Single training step"""
    
    solver.model.train()
    optimizer.zero_grad()
    
    # Boundary loss
    boundary_pred = solver.forward(boundary_data)
    boundary_input = {
        "x": boundary_data["x"],
        "y": boundary_data["y"],
        "boundary_u": boundary_data["u"],
        "boundary_v": boundary_data["v"]
    }
    boundary_losses = solver.loss(boundary_input, boundary_pred)
    
    # Interior PDE loss
    interior_pred = solver.forward(interior_data)
    interior_losses = solver.loss(interior_data, interior_pred)
    
    # Combine losses
    total_loss = boundary_losses["boundary_loss"] + interior_losses["total_loss"]
    
    # Backward pass
    total_loss.backward()
    
    # Gradient synchronization for distributed training
    if solver.dist.distributed:
        torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        total_loss /= solver.dist.world_size
    
    optimizer.step()
    
    return {
        "total_loss": total_loss.item(),
        "boundary_loss": boundary_losses["boundary_loss"].item(),
        "continuity_loss": interior_losses.get("continuity_loss", torch.tensor(0.0)).item(),
        "momentum_x_loss": interior_losses.get("momentum_x_loss", torch.tensor(0.0)).item(),
        "momentum_y_loss": interior_losses.get("momentum_y_loss", torch.tensor(0.0)).item(),
        "evm_constraint_loss": interior_losses.get("evm_constraint_loss", torch.tensor(0.0)).item(),
    }


def train_simple(
    solver: PhysicsNeMoPINNSolver,
    cfg: DictConfig,
    logger: PythonLogger
):
    """Simple training function for testing - single stage only"""
    
    stage_name = "Simple Test"
    alpha_evm = 0.01  # Fixed EVM parameter
    num_epochs = cfg.simple_training.num_epochs  # Much fewer epochs for testing
    learning_rate = cfg.simple_training.lr
    
    if solver.dist.rank == 0:
        logger.info(f"Starting {stage_name}: alpha_evm={alpha_evm}, lr={learning_rate}, epochs={num_epochs}")
    
    # Update solver parameters
    solver.set_alpha_evm(alpha_evm)
    solver.current_stage = stage_name
    
    # Create trainer components
    optimizer, scheduler, boundary_data, interior_data = create_trainer(cfg, solver)
    optimizer.param_groups[0]['lr'] = learning_rate
    
    # Training loop
    for epoch in range(num_epochs):
        
        # Simple EVM network schedule - freeze for first half, unfreeze for second half
        if epoch == 0:
            solver.freeze_evm_net()
        elif epoch == num_epochs // 2:
            solver.unfreeze_evm_net()
        
        # Training step
        losses = training_step(
            solver, boundary_data, interior_data, optimizer, epoch, cfg
        )
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # More frequent logging for testing
        if solver.dist.rank == 0 and (epoch % cfg.simple_training.log_freq == 0 or epoch == 0):
            log_str = f"Epoch {epoch+1}/{num_epochs}, Stage: {stage_name}"
            log_str += f", Total Loss: {losses['total_loss']:.3e}"
            log_str += f", Boundary: {losses['boundary_loss']:.3e}"
            log_str += f", Continuity: {losses['continuity_loss']:.3e}"
            log_str += f", Momentum X: {losses['momentum_x_loss']:.3e}"
            log_str += f", Momentum Y: {losses['momentum_y_loss']:.3e}"
            log_str += f", EVM: {losses['evm_constraint_loss']:.3e}"
            logger.info(log_str)
        
        # Less frequent checkpointing for testing
        if solver.dist.rank == 0 and epoch > 0 and epoch % cfg.simple_training.checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                cfg.checkpoint_dir, 
                f"model_simple_test_epoch_{epoch}.pth"
            )
            save_checkpoint(
                solver.model.state_dict(),
                optimizer.state_dict(),
                epoch,
                checkpoint_path
            )
            
    # Final evaluation
    if solver.dist.rank == 0:
        solver.model.eval()
        with torch.no_grad():
            # Quick evaluation on boundary data
            boundary_pred = solver.forward(boundary_data)
            boundary_input = {
                "x": boundary_data["x"],
                "y": boundary_data["y"],
                "boundary_u": boundary_data["u"],
                "boundary_v": boundary_data["v"]
            }
            final_boundary_losses = solver.loss(boundary_input, boundary_pred)
            
            # Quick evaluation on interior data
            interior_pred = solver.forward(interior_data)
            final_interior_losses = solver.loss(interior_data, interior_pred)
            
            final_total_loss = final_boundary_losses["boundary_loss"] + final_interior_losses["total_loss"]
            
            logger.info(f"Simple test completed - Final Total Loss: {final_total_loss.item():.3e}")
            logger.info(f"Final Boundary Loss: {final_boundary_losses['boundary_loss'].item():.3e}")
            
            # Save final model
            final_checkpoint_path = os.path.join(cfg.checkpoint_dir, "model_simple_test_final.pth")
            save_checkpoint(
                solver.model.state_dict(),
                optimizer.state_dict(),
                num_epochs,
                final_checkpoint_path
            )
            logger.info(f"Final model saved to {final_checkpoint_path}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config_simple")
def main(cfg: DictConfig) -> None:
    """Main simple training function for testing"""
    
    # Initialize distributed training
    DistributedManager.initialize()
    dist = DistributedManager()
    
    # Initialize loggers
    initialize_loggers()
    logger = PythonLogger("simple_train")
    
    if dist.rank == 0:
        logger.info("Starting PhysicsNeMo PINN Simple Training (Test Version)")
        logger.info(f"Distributed training: {dist.distributed}")
        logger.info(f"World size: {dist.world_size}")
        logger.info(f"Device: {dist.device}")
        logger.info("This is a simplified version for testing on dual P100 GPUs")
    
    try:
        # Initialize solver
        solver = PhysicsNeMoPINNSolver(cfg)
        
        # Load checkpoint if specified
        if cfg.checkpoint_path:
            checkpoint = load_checkpoint(cfg.checkpoint_path)
            solver.model.load_state_dict(checkpoint['model_state_dict'])
            if dist.rank == 0:
                logger.info(f"Loaded checkpoint from {cfg.checkpoint_path}")
        
        # Run simple training
        train_simple(solver, cfg, logger)
        
        if dist.rank == 0:
            logger.info("Simple training completed successfully!")
            logger.info("Ready for deployment to Dell R740 server with dual P100 GPUs")
            
    except Exception as e:
        if dist.rank == 0:
            logger.error(f"Simple training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        raise e
    
    finally:
        # Cleanup distributed training
        if dist.distributed:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()