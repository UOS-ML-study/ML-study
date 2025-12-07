# [3팀/얼굴 감정 분류]

## 1. 팀 정보
- 팀 번호: 3팀
- 팀명: 얼굴 감정 분류
- 팀원:
  - 강민석 (PM)
    - 역할: 모델링(기본 CNN 설계, 실험 플랜), 전체 흐름 및 방법론
  - 김대원 
    - 역할: 데이터(데이터 정리, 라벨 분포 분석, Augmentation 설계), EDA 및 데이터 파트
  - 김미라 
    - 역할: 모델링(하미퍼파라미터 튜닝, 여러 CNN 구조 비교), 모델 성능 비교 파트
  - 김수환 
    - 역할: 데이터(추가 데이터셋 조사/정리), LLM 활용 : 결과 설명문 자동 생성
  - 이보람 
    - 역할: 자료 정리 (데모, 데이터셋, 한계, 윤리 이슈 정리)

---

## 2. 프로젝트 개요

- 한 줄 설명:  
  얼굴 이미지를 입력받아 인간의 감정을 5개 클래스(Angry, Fear, Happy, Sad, Suprise)로 분류하는 딥러닝 기반 감정 인식 모델을 구축하는 프로젝트입니다.

- 키워드:  
  - #HumanFace #EmotionRecognition #CNN #EfficientNet #YOLOv8 #ImageClassification

---

## 3. 데이터 소개

- 출처:
  - Kaggle 공개 데이터셋: **Human Face Emotions**  
    - https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions
- 데이터 형식:
  - 감정 레이블별 폴더 구조로 정리된 얼굴 이미지
  - 주로 48×48 ~ 224×224 픽셀 범위의 정규화 가능한 그레이스케일/컬러 이미지
- 주요 컬럼(개념적으로):
  - `image_path`: 이미지 파일 경로 (예: `Angry/Angry_0001.jpg`)
  - `class`: 감정 클래스 이름 (Angry, Happy, …)
  - `label`: 감정 클래스 정수 라벨 (0 ~ 6)
- 기간:
  - 특정 수집 기간 정보는 공개되지 않았으며, 정적 이미지 데이터셋으로 사용
- 전처리 개요:
  - 모든 이미지를 **단일 크기(예: 128×128)**로 리사이즈
  - 흑백 이미지도 **RGB 3채널로 통일**
  - 픽셀 값을 [0, 1] 범위로 스케일링 후, train 셋 기준 **채널별 mean/std로 정규화**
  - 감정 클래스별 라벨 매핑(문자열 → 정수) 및 **어노테이션 정합성 검증**
  - `train/val/test` 분할 정보는 `data/processed/*.csv` 로 관리

---

## 4. 분석/모델링 목표

- 분석/모델링 질문:
  1. 얼굴 이미지 기반 감정 분류에서 **기본 CNN, EfficientNet-b0, YOLOv8n-cls** 중 어떤 모델이 더 높은 정확도와 F1-score를 보이는가?
  
  2. 감정 클래스별(Angry, Happy, Sad, 등)로 분류 성능이 어떻게 다른지, 특정 감정에서 오분류가 집중되는 패턴이 있는가?

  3. 동일한 데이터 전처리·학습 조건(epoch, batch size, 이미지 크기)을 맞췄을 때, 모델 복잡도와 성능 간 **트레이드오프**는 어떻게 나타나는가?

- 사용 방법(모델링 접근):
  - EDA:
    - 클래스 분포 확인 (감정별 데이터 수 불균형 여부)
    - 이미지 샘플 시각화 (표정, 조명, 해상도 등 특성 파악)
  - 모델링:
    - **SimpleCNN**: 직접 구현한 CNN 기반 분류 모델
    - **EfficientNet-b0**: ImageNet 사전학습 가중치를 활용한 Transfer Learning
    - **YOLOv8n-cls**: Ultralytics YOLOv8 분류 전용 모델(`yolov8n-cls`)을 활용한 감정 분류
  - 평가 지표:
    - Accuracy
    - Precision / Recall / F1-score (macro 기준)
    - Confusion Matrix(혼동 행렬) 기반 클래스별 오류 분석

---

## 5. 폴더 구조

```text
project-root/
├─ README.md
├─ requirements.txt
│
├─ config/
│  └─ parms.yaml                # 데이터 경로, 하이퍼파라미터 등 설정
│
├─ data/
│  ├─ raw/                      # 원본 얼굴 감정 이미지 (Kaggle 데이터)
│  ├─ interim/                  # 1차 전처리(리사이즈, RGB 변환 등) 결과
│  ├─ processed/                # train/val/test CSV, 최종 분석용 메타데이터
│  └─ yolo_cls/                 # YOLOv8 분류 학습용 폴더 구조(train/val/test/<class>)
│
├─ src/
│  ├─ __init__.py
│  ├─ preprocessing.py          # 이미지 전처리(크기 통일, 정규화, 어노테이션 정합성 유지)
│  ├─ data_split.py             # train/val/test 분할 스크립트
│  ├─ data_loader.py            # PyTorch Dataset/DataLoader 정의
│  ├─ features.py               # 피처/통계량 계산, EDA 보조 함수
│  ├─ modeling_cnn.py           # CNN 모델 정의 및 학습/검증
│  ├─ modeling_efficientNet.py  # EfficientNet 모델 정의 및 학습/검증
│  └─ modeling_yolo.py          # YOLOv8n-cls 분류 모델 학습 스크립트
│
├─ notebooks/
│  ├─ 01_eda.ipynb              # 데이터 구성 및 이미지 EDA
│  ├─ 02_feature_engineering.ipynb  # 간단 피처/통계, 전처리 로직 점검
│  ├─ 03_modeling_analysis.ipynb    # 3개 모델 성능 비교·분석
│  └─ 04_visualization_report.ipynb # 발표용 그래프·표 생성 및 reports 저장
│
├─ runs/
│  ├─ cnn/                      # CNN 학습 결과(체크포인트, 로그 등)
│  ├─ efficientnet/             # EfficientNet 학습 결과
│  └─ yolo_cls/                 # YOLOv8n-cls 학습 결과(exp, weights 등)
│
└─ reports/
   ├─ figures/                  # 최종 발표용 시각화(혼동행렬, 비교 그래프 등)
   └─ model_comparison_summary.csv  # 모델별 성능 요약 테이블

