# IPP_work-study

## 1. [파이토치 튜툐리얼](https://github.com/KangHongJun/IPP_Xiilab/tree/main/pytorch)

### 진행과정
 [pytorch 튜툐리얼](https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html)을 Pycharm환경에서 conda 인터프리터로 설정하여 진행했다. 

2021년도에 한국IT비즈니스진흥협회에서 고급 딥러닝 과정을 들었기에 용어가 친숙했지만, 까먹은 상태였기 때문에 용어와 함수들의 의미에 대해 인지하는 과정이었다.



## 2. [yolov5를 이용한 재난 감지](https://github.com/KangHongJun/IPP_Xiilab/tree/main/collapse_data_train)

### 진행과정
[yolov5](https://github.com/ultralytics/yolov5)로 이미지/영상에서 사람&불&건물 붕괴를 인식할 수 있도록 해봤다.

 라이브러리는 사용해봤지만, 오픈소스를 clone하여 사용해보는것은 처음이었기 때문에 requirements.txt로 환경설정하는것을 처음 해봐서 신선한 경험이었고,<br>
pytorch 튜툐리얼을 진행하던 환경에 추가로 세팅하려니 라이브러리들이 서로 버전 충돌이 나서 갈아엎으면서 진행했다.<br>
 환경세팅이 끝난 후에는 yolov5의 공식문서에서 커스텀 데이터 훈련방법을 보면서 roboflow에서 fire 데이터를 다운받고 훈련시킨 후 이미지/영상을 넣어보니 불을 잘 감지했다.
그 다음에는 사람과 건물 붕괴 데이터도 넣어서 학습해봤지만, 컴퓨터 성능문제로 훈현을 조금 밖에 돌리지 못하여 성능이 좋지 않았다.




## 3. [Origin-NMS](https://github.com/KangHongJun/Origin-NMS)

### 진행과정
 먼저 오픈소스 [sahi](https://github.com/obss/sahi)를 detect model을 yolov5로 하여 세팅했다.
 [sahi라는 오픈소스의 목적과 문제점에 대해 파악하고 내장되어 있는 merge 알고리즘 NMS. NMM. GREEDYNMM을 하나씩 돌려보고 정리했다.](https://github.com/KangHongJun/IPP_Xiilab/tree/main/sahi_improve/search)<br>
 
