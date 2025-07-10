# PuDDing

**PuDDing 모델**

본 프로젝트는 **Prompt-based Dynamic Depth pruning (PuDDing)** 모델을 기반으로, 아래의 네 단계로 구성되어 있습니다:

1. **Omission layer 선정**
2. **Dataset 생성**
3. **Router training**
4. **Trained router를 사용한 pruned model evaluation**

현재 코드는 간단히 실행할 수 있는 형태로 정리되어 있으며, 향후 더 깔끔하고 효율적인 구조로 업데이트할 예정입니다.

---

# 설치 방법

`environment.yml` 파일을 참고하여 conda 환경을 생성하세요.


# 사용 방법


💻 사용 방법
1. **Omission layer 선정**

    1-a. 아래 스크립트 실행
    - bash _1_layerset_llama.sh
    
    1-b. result/9_llama/data/4_analysis 폴더에 omission set들이 저장됩니다.

    1-c. 생성된 10개의 omission set 결과를 확인 후, 이후 dataset 생성을 위해 CSV 파일로 저장해야 합니다.
    - 예시 파일: codes/llama_layer_list_6_advanced_tasks.csv
    - omission set 결과를 예시와 같은 형식의 csv 파일로 변환하여 저장하세요.


2. **Dataset 생성**
    2-a. 아래 스크립트 실행
    - bash _2_dataset_llama.sh
    - 새로운 omission set을 선정한 경우, 해당 csv 파일을 2_dataset_llama.sh 스크립트 내 open_path 위치에 지정해야 합니다.
    - 또한 v23_script 내부의 with open('codes/llama_layer_list_6_advanced_tasks.csv', mode="r") 부분도 새로운 오메션셋 리스트 csv 파일 경로로 수정 필요합니다.

    2-b. result/9_llama/data/5_log5 폴더에 task별 dataset이 생성됩니다.

    2-c. 생성된 5개 task별 dataset을 하나의 csv 파일(all_log.csv)로 통합해야 합니다.
    - 예시 파일: result/9_llama/data/6_adavanced_tasks/all_log.csv


3. **Router training**
    3-a. 아래 스크립트 실행
    - bash _3_train_router_MSE.sh
    - 새로운 dataset을 생성한 경우, _3_train_router_MSE.sh 스크립트의 csv file path를 해당 dataset으로 수정해야 합니다.
    - 현재는 예시 파일을 기반으로 training하도록 설정되어 있습니다.

4. **Eval router**
    4-a. 아래 스크립트 실행
    - bash _4_eval_router.sh

-----------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------



