import torch
import torch.optim as optim
from data.scheduler import CosineAnnealingRestartLR, GradualWarmupScheduler

def test_manual_scheduler_reconstruction():
    print("=== Manual Scheduler Reconstruction Test ===")
    print("Testing with manual scheduler reconstruction (no state_dict)")
    print()
    
    # 설정값들 (options.py와 동일)
    division = 4
    lr_initial = 1e-4 / division  # 2.5e-5
    total_epochs = 1500
    warmup_epochs = 3
    eta_min = 1e-7
    
    print(f"Settings:")
    print(f"  - Initial LR: {lr_initial}")
    print(f"  - Total epochs: {total_epochs}")
    print(f"  - Warmup epochs: {warmup_epochs}")
    print(f"  - Eta min: {eta_min}")
    print()
    
    # 공통 모델 템플릿
    model_template = torch.nn.Linear(10, 1)
    
    print("=" * 80)
    print("=== 첫번째: 1500 epoch 연속 훈련 (1000 epoch에서 상태 저장) ===")
    
    # 첫번째: 연속 훈련
    model1 = torch.nn.Linear(10, 1)
    model1.load_state_dict(model_template.state_dict())
    optimizer1 = optim.Adam(model1.parameters(), lr=lr_initial)
    
    # Scheduler 설정
    cosine_scheduler1 = CosineAnnealingRestartLR(
        optimizer=optimizer1, 
        periods=[total_epochs - warmup_epochs],  # [1497]
        restart_weights=[1],
        eta_min=eta_min
    )
    
    scheduler1 = GradualWarmupScheduler(
        optimizer=optimizer1,
        after_scheduler=cosine_scheduler1,
        multiplier=1,
        total_epoch=warmup_epochs
    )
    
    print(f"Cosine scheduler periods: {cosine_scheduler1.periods}")
    print()
    
    # 1500 epoch 훈련
    torch.manual_seed(12345)
    
    checkpoint_data = None
    final_lr_continuous = None
    
    for epoch in range(total_epochs):
        scheduler1.step()
        current_lr = optimizer1.param_groups[0]['lr']
        
        # Dummy training step
        dummy_input = torch.randn(5, 10)
        dummy_target = torch.randn(5, 1)
        optimizer1.zero_grad()
        output = model1(dummy_input)
        loss = torch.nn.functional.mse_loss(output, dummy_target)
        loss.backward()
        optimizer1.step()
        
        # 1000 epoch에서 필요한 상태만 저장
        if epoch + 1 == 1000:
            print(f"Epoch {epoch+1}: LR = {current_lr:.16e} - SAVING CHECKPOINT")
            
            checkpoint_data = {
                'model_state_dict': model1.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'epoch': epoch + 1,
                'lr': current_lr
            }
            
        # 주요 epoch에서 LR 출력
        if epoch + 1 in [1, 100, 500, 999, 1000, 1001, 1400, 1500]:
            print(f"Continuous - Epoch {epoch+1}: LR = {current_lr:.16e}")
    
    final_lr_continuous = optimizer1.param_groups[0]['lr']
    print(f"\n첫번째 실험 완료 - Final LR (epoch 1500): {final_lr_continuous:.16e}")
    
    print("\n" + "=" * 80)
    print("=== 두번째: 1000 epoch 체크포인트에서 복원 후 1500까지 훈련 ===")
    
    # 두번째: 체크포인트 복원
    model2 = torch.nn.Linear(10, 1)
    optimizer2 = optim.Adam(model2.parameters(), lr=lr_initial)
    
    # 체크포인트 로드
    print("Loading checkpoint...")
    model2.load_state_dict(checkpoint_data['model_state_dict'])
    optimizer2.load_state_dict(checkpoint_data['optimizer_state_dict'])
    
    loaded_lr = optimizer2.param_groups[0]['lr']
    saved_lr = checkpoint_data['lr']
    print(f"Saved LR: {saved_lr:.16e}")
    print(f"Loaded LR: {loaded_lr:.16e}")
    print(f"LR match: {loaded_lr == saved_lr}")
    
    
    # Scheduler를 정확한 last_epoch로 재구성
    start_epoch = checkpoint_data['epoch']
    
    print(f"Rebuilding scheduler from epoch {start_epoch}...")
    # 올바른 방식: 생성 시점에 last_epoch 전달
    cosine_scheduler2 = CosineAnnealingRestartLR(
        optimizer=optimizer2, 
        periods=[total_epochs - warmup_epochs],  # [1497]
        restart_weights=[1],
        eta_min=eta_min,
        last_epoch=start_epoch - 1 - warmup_epochs  # warmup 이후의 epoch 수
    )
    
    scheduler2 = GradualWarmupScheduler(
        optimizer=optimizer2,
        after_scheduler=cosine_scheduler2,
        multiplier=1,
        total_epoch=warmup_epochs
    )
    
    # GradualWarmupScheduler는 생성 후 last_epoch 설정이 더 정확
    scheduler2.last_epoch = start_epoch - 1
    
    print(f"Scheduler last_epoch: {scheduler2.last_epoch}")
    print(f"Cosine scheduler last_epoch: {cosine_scheduler2.last_epoch}")
    print(f"Resuming from epoch {start_epoch}")
    
    # 첫 번째 step에서 LR 확인
    print("\nFirst step after resume:")
    scheduler2.step()
    current_lr_after_step = optimizer2.param_groups[0]['lr']
    print(f"LR after first step: {current_lr_after_step:.16e}")
    
    # 1001부터 1500까지 훈련 (첫 step은 이미 했으므로 start_epoch+1부터)
    for epoch in range(start_epoch + 1, total_epochs):
        scheduler2.step()
        current_lr = optimizer2.param_groups[0]['lr']
        
        # Same dummy training step
        dummy_input = torch.randn(5, 10)
        dummy_target = torch.randn(5, 1)
        optimizer2.zero_grad()
        output = model2(dummy_input)
        loss = torch.nn.functional.mse_loss(output, dummy_target)
        loss.backward()
        optimizer2.step()
        
        if epoch + 1 in [1001, 1400, 1500]:
            print(f"Resumed - Epoch {epoch+1}: LR = {current_lr:.16e}")
    
    final_lr_resumed = optimizer2.param_groups[0]['lr']
    print(f"\n두번째 실험 완료 - Final LR (epoch 1500): {final_lr_resumed:.16e}")
    
    print("\n" + "=" * 80)
    print("=== 결과 비교 ===")
    
    print(f"첫번째 (연속 훈련) epoch 1500 LR: {final_lr_continuous:.16e}")
    print(f"두번째 (체크포인트 복원) epoch 1500 LR: {final_lr_resumed:.16e}")
    
    difference = abs(final_lr_continuous - final_lr_resumed)
    print(f"차이 (절댓값): {difference:.16e}")
    
    # 정확도 체크
    perfect_match = difference == 0.0
    very_close = difference < 1e-15
    close_enough = difference < 1e-10
    
    print(f"완벽한 일치 (차이 = 0): {perfect_match}")
    print(f"매우 근접 (차이 < 1e-15): {very_close}")
    print(f"충분히 근접 (차이 < 1e-10): {close_enough}")
    
    if perfect_match:
        print("🎉 100% 완벽한 복원 성공!")
    elif very_close:
        print("✅ 실질적으로 완벽한 복원 (부동소수점 정밀도 한계)")
    elif close_enough:
        print("✅ 실용적으로 완벽한 복원")
    else:
        print("❌ 복원에 문제가 있습니다")
    
    return {
        'continuous': final_lr_continuous,
        'resumed': final_lr_resumed,
        'difference': difference,
        'perfect_match': perfect_match
    }

if __name__ == "__main__":
    result = test_manual_scheduler_reconstruction()
    print(f"\nFinal Result: Perfect restoration = {result['perfect_match']}")