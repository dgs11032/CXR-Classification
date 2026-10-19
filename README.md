# P1 CXR Pneumonia Classification

계층적 멀티태스크 학습과 지도 대조학습을 이용해 소아 흉부 X-ray를 `NORMAL`, `BACTERIAL_PNEUMONIA`, `VIRAL_PNEUMONIA`로 분류하고 특징 표현과 예측 신뢰성을 분석하는 프로젝트입니다.

## 연구 설계

모든 실험은 같은 ConvNeXt-Tiny backbone과 데이터 분할을 사용합니다.

| 실험 | 구조 | 지도 대조학습 |
|---|---|---|
| E0 | Flat 3-class | 미적용 |
| E1 | Flat 3-class | 적용 |
| E2 | Hierarchical detection/subtype | 미적용 |
| E3 | Hierarchical detection/subtype | 적용 |

현재 Day 1 범위는 데이터 검증·분할과 E0 baseline입니다.

## 데이터

- 원본 S3: `s3://aws-say3-dataset1/pneumonia-xray/chest_xray/`
- 기본 로컬 루트: `chest_xray/chest_xray/`
- 공식 test 624장은 최종 모델과 설정을 확정하기 전까지 평가에 사용하지 않습니다.
- 기존 16장 validation은 사용하지 않고 원본 train에서 파일명 기반 proxy-group 85:15 train/validation을 생성합니다. 실제 환자 ID가 아니므로 patient-level split이라고 주장하지 않습니다.

AWS 인증 후 전체 데이터를 다음 명령으로 일괄 동기화할 수 있습니다. 현재 공식 데이터는 이 경로에 동기화되어 있으며 S3와 로컬의 5,856개 파일 및 1,236,482,806바이트가 일치함을 확인했습니다.

```bash
aws login
aws s3 sync s3://aws-say3-dataset1/pneumonia-xray/chest_xray/ chest_xray/chest_xray/ --region ap-northeast-2
```

## 검증 실행

워크스페이스 공용 가상환경의 필수 패키지를 사용합니다. 현재 단계별 검증 명령은 다음과 같습니다.

```bash
cd P1_CXR
../.venv/bin/python scripts/audit_dataset.py
../.venv/bin/python scripts/create_splits.py
../.venv/bin/python scripts/eda_images.py
../.venv/bin/python scripts/validate_data_pipeline.py
../.venv/bin/python scripts/validate_e0_dryrun.py
```

전체 E0 학습 진입점은 실수로 실행되지 않도록 명시적 승인 플래그가 필요합니다. 현재 STEP 7에서는 실행하지 않습니다.

```bash
# 사용자 승인 후 다음 단계에서만 실행
../.venv/bin/python scripts/train.py \
  --config configs/e0_training.yaml \
  --confirm-full-training
```

## 현재 공식 산출물

- 데이터 감사: `artifacts/data_audit/`
- 고정 분할: `artifacts/splits/`
- 이미지 EDA: `artifacts/eda/`
- 데이터 파이프라인 검증: `artifacts/data_pipeline/`
- E0 Dry-run: `artifacts/experiments/e0_dryrun/`

> 본 프로젝트는 교육·연구용이며 실제 의료 진단을 대체하지 않습니다.
