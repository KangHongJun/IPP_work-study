# IPP_work-study

## 1. [파이토치 튜토리얼](https://github.com/KangHongJun/IPP_Xiilab/tree/main/pytorch)

### 진행과정
 [pytorch 튜툐리얼](https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html)을 Pycharm환경에서 conda 인터프리터로 설정하여 진행했다. 

2021년도에 한국IT비즈니스진흥협회에서 고급 딥러닝 과정을 들었기에 용어가 친숙했지만, 까먹은 상태였기 때문에 용어와 함수들의 의미에 대해 인지하는 시간이었다.



## 2. [yolov5를 이용한 재난 감지](https://github.com/KangHongJun/IPP_Xiilab/tree/main/collapse_data_train)

### 진행과정
[yolov5](https://github.com/ultralytics/yolov5)로 이미지/영상에서 사람&불&건물 붕괴를 인식할 수 있도록 해봤다.

 라이브러리는 사용해봤지만, 오픈소스를 clone하여 사용해보는것은 처음이었기 때문에 requirements.txt로 환경설정하는것을 처음 해봐서 신선한 경험이었고,<br>
pytorch 튜툐리얼을 진행하던 환경에 추가로 세팅하려니 라이브러리들이 서로 버전 충돌이 나서 갈아엎으면서 진행했다.<br>
 환경세팅이 끝난 후에는 yolov5의 공식문서에서 커스텀 데이터 훈련방법을 보면서 roboflow에서 fire 데이터를 다운받고 훈련시킨 후 이미지/영상을 넣어보니 불을 잘 감지했다.
그 다음에는 사람과 건물 붕괴 데이터도 넣어서 학습해봤지만, 컴퓨터 성능문제로 훈련을 조금 밖에 돌리지 못하여 성능이 좋지 않았다.
<img src="https://github.com/KangHongJun/IPP_Xiilab/blob/main/collapse_data_train/val_batch0_pred.jpg">



## 3. [Origin-NMS](https://github.com/KangHongJun/Origin-NMS)(Sahi에 내장된 병합 알고리즘 NMS개선)

### 진행과정

 먼저 오픈소스 [sahi](https://github.com/obss/sahi)를 detect model을 yolov5로 하여 세팅했다.
 [sahi라는 오픈소스의 목적과 문제점에 대해 파악하고 내장되어 있는 merge 알고리즘 NMS. NMM. GREEDYNMM을 하나씩 돌려보고 정리했고](https://github.com/KangHongJun/IPP_Xiilab/tree/main/sahi_improve/search) (sahi를 이용할때 겹치는 박스가 삭제되지 않거나 올바른 box까지 삭제하는 상황 개선)<br>
정리한 것을 바탕으로 피드백을 받아서 sahi의 문제점을 개선하기 위해 다음의 단계를 거쳤다.<br>
1. merge연산 진행시 IOU연산이 아닌 다른 연산을 하는 눈문에 대한 코드를 탐색해봤지만 공개한 것이 없어서 찾기 힘들었기 때문에 겹침정도를 연산하는 IOU와 IOS에 대해 이해하고 의논후에 NMS와IOS기반으로 개선해나가는것으로 방향을 잡았다.<br>
2. NMS-IOS를 진행하면 사라지는 box가 너무 많기 때문에 오리지널 box를 복구하는 방향으로 가자고 피드백을 받아 코드를 작성하고 돌려보니 box를 복구한 만큼 겹치는 부분이 많아졌기 때문에 그것을 해소할 방법이 필요했다. <br>
3. 겹치는 box를 삭제하기 위해 nms 진행시 최고score가 아닌 오리지널 box score기준으로 nms를 진행해봤지만 사소한 변화만 있었다.
4. 지금까지 돌려본 결과들을 살펴보며 고민끝에 슬라이싱 이미지의 box끼리 먼저 nms를 하고 그 다음에 오리지널 이미지의 box를 합쳐서 nms를 하는 방식으로 해보자는 피드백을 받고 진행시켰더니 괜찮은 결과물을 얻어낼 수 있었다.

#### 느낀점
 코드를 개선해 나가기 위해서는 오픈소스의 구조와 가공하기 위한 반환 타입 등을 알아야 했기 때문에 처음으로 오픈소스를 디버깅하면서 분석해봤는데 어려운 만큼 코드 짜는 방식이나 문법에 대한 이해가 깊어진다고 느꼈고, 많은 도움을 받아서 진행한 만큼 누군가의 피드백을 받아가며 개선해나가는 과정이 재미있었다.
 코드를 개선해 나가면서 문법에 혼동이 있어서 코드를 잘못 작성한다거나, 결과 이미지 이름을 애매하게 작성하여 나중에 검토 할 때 어떤 코드에서 나온 결과물인지 모르는 상황이 나왔기 때문에 버전 관리 및 결과물에 대한 기록작성에 대한 중요성을 많이 느꼈고, sahi-yolox에 이 알고리즘을 적용해볼때는 오픈소스가 여러개 엮이니 환경세팅이 복잡했다.(pycocotools/c++기반/conda로 설치)  문서도 꼼꼼히 읽고, 그때그때 가상환경을 만들어서 해야겠다고 느꼇다.
 
 <p float="left">
    <div align = "center">
       <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/NMS_yolov5m.png"><br>
     [ sahi를 이용한 이미지 detect 후 NMS ]<br><br>
       <img src="https://github.com/KangHongJun/Origin-NMS/blob/main/Images/Origin_NMS_yolov5m.png"><br>
     [ 개선한 방법 ]
    </div>
  </p>
  기존 NMS와 비교하면 상당히 개선된 것을 확인할 수 있다.


