# annotool
1. download yolov4.weights from "https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view"
2. put yolov4.weights in pjtlibs
3. install packages in requirements.txt
4. build qt.py (cuda 10.1, cudnn 7.6.5, ubuntu 18.04)


![스크린샷, 2020-12-04 16-43-30](https://user-images.githubusercontent.com/70502705/101143621-81594580-365a-11eb-9bcf-f81cfa04b7a7.png)

= 위젯 구성 =
* 각 버튼의 숏컷들은 버튼 괄호안에 표시되어 있습니다
* 화면 가운데 메인 프레임에 영상이 표시됩니다
* 우측상단 프레임에 저장된 사진과 프레임번호, 라벨이 표시됩니다
* 우측 위젯에는 기록한 프레임과 라벨이 표시됩니다

= 버튼 상세 =
1) File : 영상 불러오기 UI제공
2) Load : 오브젝트를 설정하기 위한 첫번째 프레임을 출력합니다
3) Object : 추적할 오브젝트 입니다. 폴더가 생성됩니다
4) Target: 영상에서 기록할 타겟박스의 넘버입니다 타겟이 바뀌어도 같은 오브젝트폴더에 저장됩니다
추적하던 오브젝트의 타겟박스가 바뀌었을때 사용합니다
5) Track start : 영상이 재생됩니다
6) Play & Pause : 영상을 일시정지시키거나 재생시킬 수 있습니다
7) 화살표 : 배속을 늘리거나 줄입니다 1배속부터 정수배가 가능합니다
8) Make json : 위에 기록된 리스트가 현재 오브젝트 폴더에 json 파일로 저장됩니다
9) Delete : 리스트에서 선택된 항목을 제거합니다. Make json을 해줘야 파일에 반영됩니다
10) Action start : 액션의 시작과 종료를 기록하기 위한 토글키입니다
11) Show target only : 추적중인 박스(녹색)만 표시하는 토글키입니다
12) Open folder : 저장되는 폴더를 엽니다
13) Reset : 영상을 종료하고 초기화합니다


* TODO
 json 파일을 불러와 수정할 수 있게 하기  
 다른 영상의 오브젝트 구별하기 위해 영상 이름으로 폴더 만들어 관리하기  
 지웠던 리스트 항목 복구기능  
 fps 부스트 또는 트래킹정보 저장  
