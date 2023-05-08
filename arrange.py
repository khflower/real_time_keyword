import cv2
import numpy as np

def crop_blank(img):

    ''' 도형의 외곽선을 따라 둘러싸는 사각형 좌표를 반환하는 cv2의 boundingRect() 함수를
    이용한 인물 주변의 빈 공간을 잘라주는 함수 \n
    input : img / image array 한장 \n
    output : image array 한장  \n'''

    # 외곽선을 따기 위해 컬러 정보를 없애고 boundingRect를 하여 둘러싸는 사각형 정보를 가져와 그 부분만 잘라줌.
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY);
    index = cv2.boundingRect(gray);
    (x,y,w,h)=index;
    img_crop = img[y:y + h + 1, x : x + w + 1, :];


    return img_crop

def resize_after_crop(img_crop_list, ground_size=[400, 600] ,rat=0.9):

    ''' image의 인물 주위 공백을 잘라낸 이후, 각 이미지의 순위에 맞게 이미지의 사이즈를 조절하는 함수 \n
    input : img_crop_list / 얼굴 사이즈에 맞게 image array가 포함된 list \n
            gound_size / 배경 사이즈를 나타내는 리스트 [가로, 세로 ] \n
    ouput : list 안의 image array를 index 순서에 맞게 크기를 조절한 뒤 다시 하나로 묶은 list \n '''

    img_size = [];
    img_resize = [];

    # 각 image array의 픽셀 수(크기) 저장
    for i in range(len(img_crop_list)):
        img_size.append(img_crop_list[i].shape[0] * img_crop_list[i].shape[1]);

    # image array를 전체 배경의 1/7 크기를 기준으로 resize 하기 위한 저장 변수
    stand = 1 / 7 * ground_size[0] * ground_size[1];

    for i in range(len(img_crop_list)):

        # 가로길이 > 3*세로 길이 // 길쭉한 로고는 배경 밖으로 튀어나가 오류가 생겨서 제일 낮은 순위와 동일 사이즈가 되는 축소 비율
        if img_crop_list[i].shape[1] > 3 * img_crop_list[i].shape[0]:
            temp_ratio = np.sqrt((rat ** 10) * stand / img_size[i]);

        # 위의 경우 제외 일반적인 image array는 list 안에서의 index(순위)에 맞게 rat(순위 별 이미지 크기 차이)의 순위제곱을 해주기 위한 축소 비율
        else:
            temp_ratio = np.sqrt((rat ** i) * stand / img_size[i]);

        # 각 비율을 곱한 사이즈로 image array를 resize 후 저장
        img_resize.append(cv2.resize(img_crop_list[i], [round(temp_ratio * img_crop_list[i].shape[1]),
                                                   round(temp_ratio * img_crop_list[i].shape[0])]));
        img_resize[i].astype('uint8');

    return img_resize

def mapping_after_resize(img_resize_list, ground_size=[400, 600],ground_region=[0.05,0.95,0.25,0.85]):

    '''  image array를 배치에 사용할 평가함수에 응용하기 위해서 2차원 평면에 픽셀값을 가지는 곳은 1,
    여백 부분은 0으로 만든 뒤, 사람 머리가 위치한 윗 부분(세로 비율 중 위 50%)는 10으로 가중치를 부여하는 함수  \n
    (배경의 경우, 중앙 부분을 -1로 설정. 배치 시 -1을 없애는 방향으로 하여 중앙 부위에서 이미지가 퍼지도록 의도함) \n
    input : img_resize_list / 각 순위에 맞게 resize를 거친 image array가 담긴 list(즉, resize_after_crop 함수를 통해 처리된 list) \n
            ground_size / 배경 사이즈를 나타내는 리스트 [가로, 세로 ] \n
            ground_region / 배경 점수 이미지 생성 시, -1을 넣어줄 영역(가로 세로 1을 기준으로) [x_시작, x_끝, y_시작, y_끝]
    output : image array가 나타내는 인물의 모양에 맞게 1과 10으로 픽셀값이 바뀐 array가 담긴 list(decision_position 함수에 사용될 예정) \n
    '''

    img_map = []; #결과로 반환될 image array를 담을 list

    # 각 image array가 (y size, x size, z size)로 되어 있는 것을 (y size, x size)로 바꾼 뒤,
    # 0보다 큰 값의 pixel은 True, 0 혹은 음의 값을 가지면 False가 된 array를 uint8으로 타입 변경하여 0 과 1로 바꾼 뒤
    # 위에서 부터 세로의 절반을 10으로 바꿈.
    for i in range(len(img_resize_list)):
        temp = img_resize_list[i].transpose((2, 0, 1));
        temp = sum(temp);
        temp_2 = temp > 0;
        temp_2 = temp_2.astype('uint8');
        # 가로길이 > 2*세로 길이 // 길쭉한 로고 가중치를 조금 다르게 줌.

        if temp_2.shape[1] > 2 * temp_2.shape[0]:
            temp_2 = 10 * temp_2;
            # 1등과 10등을 제외하고
            # 왼쪽 편에 오게 될 이미지 중 길쭉 로고는 가로 길이의 1/4 만큼의 왼쪽 부분도 10으로 가중치를 줌.
            if i % 2 ==0 and i !=0:
                temp_2[round(0.5 * temp_2.shape[0]):, : int(temp_2.shape[1] * 2 / 3)] = 1 / 10 * temp_2[round(
                    0.5 * temp_2.shape[0]):, : int(temp_2.shape[1] * 2 / 3)];

            # 오른쪽 편에 오게 될 이미지 중 길쭉 로고는 가로 길이의 1/4 만큼 오른쪽 부분도 10으로 가중치를 줌.
            elif i % 2 ==1 and i !=9:
                temp_2[round(0.5 * temp_2.shape[0]):, int(temp_2.shape[1] * 1 / 3):] = 1 / 10 * temp_2[round(
                    0.5 * temp_2.shape[0]):, int(temp_2.shape[1] * 1 / 3):];

            else:
                pass
        # 길쭉로고가 아닌 로고와 사람은 세로의 절반 윗부분을 모두 10으로 가중치.
        else:
            temp_2[0:round(0.5 * temp_2.shape[0]), :] = 10 * temp_2[0:round(0.5 * temp_2.shape[0]), :];
        img_map.append(temp_2);

    # 배경 size의 맵 생성 후 중앙부분 -1로 설정
    temp = np.zeros((ground_size[1], ground_size[0]));
    temp[int(ground_region[2]*ground_size[1]):int(ground_region[3]*ground_size[1]), int(ground_region[0]*ground_size[0]):int(ground_region[1]*ground_size[0])] = -1;
    img_map.append(temp);

    return img_map

def init_position(ground_size=[400, 600]):

    '''  image의 첫 위치 좌표/ image의 중앙에 배경에서 위치할 index를 지정하는 함수 \n
    input : ground_size / 이미지의 처음 중앙 좌표 위치를 계산 할 배경 사이즈 [가로, 세로] \n
    ouput : 10 개의 위치 좌표(type tuple)를 포함한 list
    '''

    # 배경 이미지의 중앙 인덱스
    c_x, c_y = round(0.5 * ground_size[0]), round(0.5 * ground_size[1]);

    init_index = [(c_x, int(1.6 * c_y)), (int(0.8 * c_x), int(1.35 * c_y)), (int(1.2 * c_x), int(1.25 * c_y)), (int(0.9 * c_x), int(1.1 * c_y)),
                  (int(1.1 * c_x), int(1.0 * c_y)), (int(1.2 * c_x), int(0.85 * c_y)), (int(0.8 * c_x), int(0.75 * c_y)), (int(0.9 * c_x), int(0.6 * c_y)),
                  (int(1.1 * c_x), int(0.5 * c_y)), (c_x, int(0.35 * c_y))];


    return init_index

def func_valid(img_paddingmap_list):

    ''' 각 이미지가 중앙 부분에서 너무 서로 멀어지지 않고 적당히 떨어질 때 좋은 점수를 가져오는 평가 함수 \n
    input : img_paddingmap_list / 1과 10으로 픽셀 값이 바뀐 이미지 맵이 각 index에 맞게 배경 size에 맞추어 0으로 패딩된 이미지 array가 담긴 list \n
    output : 평가 함수의 점수, 점수가 계산되기 전 픽셀값을 가지고 있는 하나의 맵 \n
    '''
    # 겹겹히 쌓은 image를 각 좌표마다 더해줌.
    final_map = sum(img_paddingmap_list);

    # value 할 떄 dict의 카운트가 0이 되면 오류로 인해 방지용/ 가장 모서리에는 이미지가 오지 않을 거라 판단하여 항상 0일테니 하나씩만 넣어둠.
    final_map[0, 0] = -1;
    final_map[0, 1] = 22;
    final_map[1, 0] = 21;
    final_map[1, 1] = 20;
    final_map[0, 2] = 19;
    final_map[2, 0] = 30;
    final_map[2, 1] = 29;

    # 더해진 하나의 맵에서 픽셀 값(num)마다 그에 해당하는 픽셀값의 수(count)
    num, counts = np.unique(final_map, return_counts=True);
    num_counts = dict(zip(num, counts));
    value = (-3) * num_counts[-1] + (-2) * (num_counts[19] + num_counts[20]) + (-3) * (num_counts[21] + num_counts[22])+(-10**3)*(num_counts[30]+num_counts[29]);
    # 기본적으로 평가지표는 음의 값을 가지며 0에 가까울 수록 좋다고 평가됨.
    # -1은 이미지 맵 중 세그맨테이션 이미지에는 존재하지 않으며 배경 맵 중앙에만 존재하며 좋은 점수를 받기 위해 -1을 없애야 하며 -1 구역을 채우기 위해 이미지는 중앙에서 점점
    # 밖으로 이동할 것을 의도하였음.(아예 끝으로 가는 것을 막기 위해 중앙 부분만 -1로 맵핑한 것)
    # 세그맨테이션 맵에서 윗부분은 10 아랫부분은 1로 픽셀값을 주어, 19는 10+10+(-1)이 만나서, 20은 10+10이 만나 생성되며
    # 10으로 된 부분은 겹치지 않고 화면에서 보였으면 하는 부위라 이 값이 존재하는 픽셀 당 -2를 부여.
    # 3개의 이미지가 겹칠 때에는 20(10 10 1 -1) 21(-1 10 10 1) 30(10 10 10) 29(10 10 10 -1)가 생길 수 있어 29와 30에 해당하는 픽셀에는 -10^3 값 부여
    # 3개가 겹치는 건 절대 안 좋을 것이라 판단
    # 4개 이상은 발생하지 않을거라 판단하고 더 이상 추가하지 않음.


    return  value, final_map

def neighs(cur_state, walk=15 ,more=True):

    ''' 현재 상태에서 10개의 이미지 중, 1위와 10위를 제외한 이미지의 좌표 list 중 하나의 좌표만을 좌우로는 +-walk(변수 크기), 상하로는 +-walk/2 움직인
        좌표들을 반환하는 함수 \n
        input : cur_state/ 현재 상태의 index list
                more/ Neighs를 사용할 함수(decision_position)에서  False로 미리 지정하고 시작하여 False인 경우, 좌우로 이동하는 list만,
                walk / 현재 상태가 움직일 때, 이동하는 보폭으로 기본 설정 12 \n
                True인 경우에는 상하로 +-5한 list들까지 모두 반환하게 해줄 변수 \n
        output : 현재 상태에서 좌우로 +-10 혹은 상하로 +-5된 index list를 포함하는 list \n
        '''

    # output을 담을 빈 list
    neighs=[];

    # 1위와 10위를 제외한 index 중 하나에만 +-10을 하기 위한 for 문으로 1위에 해당하는 cur_state[0]과 cur_state[-1]을 제외하기 위해
    # range를 1부터 len(cur_state)-1로 설정하였음.
    # neighbor는 1위와 10위를 제외한 결과로, 8장의 이미지만 들어오면 8위는 10위가 아니므로 꼴등 이미지이지만 움직일 수 있게 하였음.

    for i in range(1,len(cur_state)):
        if i ==9:
            break
        temp1 =cur_state.copy();
        temp2 = cur_state.copy();
        temp1[i] = (cur_state[i][0] + walk , cur_state[i][1]);
        temp2[i] = (cur_state[i][0] - walk , cur_state[i][1]);
        neighs.append(temp1);
        neighs.append(temp2);

    # more 가 True인 경우에만, 상하로 index 중 하나만 +-5된 list들을 output list에 추가하도록 함.
    if more is True:
        for u in range(1, len(cur_state)):
            if u==9:
                break
            temp1 = cur_state.copy();
            temp2 = cur_state.copy();
            temp1[u] = (cur_state[u][0] , cur_state[u][1] + int(walk/2) );
            temp2[u] = (cur_state[u][0] , cur_state[u][1] - int(walk/2) );
            neighs.append(temp1);
            neighs.append(temp2);

    else:
        pass

    return neighs

def find_bestneighbor(img_map_list, neis,init_state):

    ''' 매핑된 image list와 현재 상태에서의 이웃 상태들의 list를 받아 각 상태의 좌표별로
    배경 사이즈에 맞게 매핑 이미지를 패딩하고 평가 함수를 통해 점수를 계산한 뒤, 가장 좋은 점수를 가진 상태와 좌표를 반환하는 함수 \n
    input : img_map_list / 매핑된 image list(배경 사이즈로 패딩되기 전)
            neis / cur_state의 Neighs output인 이웃 list \n
    output : 제일 점수가 높았던 이웃의 점수와 상태(각 이미지 별 배경에서 중앙좌표 list) \n
    '''

    # 이웃 상태들의 점수를 받을 빈 list
    neigh_score=[];

    for i in range(len(neis)):


        #y좌표가 처음보다 60이상 커지거나 작아지면 그만 움직이게 하려고 만든 함수수
        warm=False;
        y_inter=[];
        for j in range(len(neis[i])):
            y_inter.append(init_state[j][1]-neis[i][j][1])
        if max(y_inter) >=60 or min(y_inter)<=-60:
            warm=True;

        # 매핑된(패딩 이전의) 이미지 list와 이웃 상태를 모아놓은 list에서 하나의 상태씩 꺼내와서
        # 상태의 좌표들에 해당하게 이미지 list를 패딩해주고 이를 평가한 뒤, 점수를 list에 넣어줌.
        try:
            img_paddingmap_list=padding_position(img_map_list, neis[i]);
            value, map=func_valid(img_paddingmap_list);
            if warm:
                value=-10**10;

            else:
                pass


        # 이때, 이웃 상태에 해당하는 좌표에서 이미지를 넣었을 때, 배경의 바깥으로 이미지가 나가버리는 경우가 생기고
        # ValueError가 발생하여 이 경우에는 점수를 10의 10승으로 부여하여 절대 선택되지 않게 설정함.
        except ValueError:
            value= -10**10;
        neigh_score.append(value);

    # 점수가 제일 좋았던 상태를 best_neigh로 지정.
    best_neigh=neis[neigh_score.index(max(neigh_score))];
    return best_neigh, max(neigh_score)

def decision_position(img_map_list,trace=False,more=True,walk=15,maxRounds=80):

    ''' 매핑된 이미지 리스트를 시작 상태부터 시작하여 while문을 통해 더 높은 평가 함수의 점수를 가지는 상태를 찾아내는 함수 \n
    input : img_map_list / 매핑된(현재 상태의 좌표들로 패딩되기 전) 이미지 리스트로 함수 내에서 패딩 함수 사용하여 패딩함.
            trace / while문을 반복할 때마다 찾아서 이동한 상태와 움직인 횟수를 출력할 수 있게 하는 변수로 trace=True가 되면 출력함. \n
            more / Neighs(cur_state, more)의 more를 미리 False로 지정. 상하로의 움직임 또한 포함하려면 more=True를 넣어주어야 함. \n
            walk / neighs 함수에 들어가야 할 요소로, neighs에서 이미지 한장을 움직이는 보폭을 의미함. 기본 설정 20 \n
            maxRounds / while 문을 실행할 횟수의 제한이며 while문의 실행 1번은 현재 상태에서 이웃 상태들 중 더 좋은 상태로 현재 상태의 이동을 의미. \n
    output : while 문을 통해 찾아낸 가장 평가 점수가 높은 상태(이미지 중앙 좌표) list
    '''


    init_index=init_position();

    # 10개의 검색어에서 이미지를 전부 가져오지 못한 경우 대비한 부분으로, 가져온 이미지의 수(len(img_list))만큼
    # 현재 상태에서 제일 마지막 이미지의 y좌표와 첫번 째 y좌표의 중앙 값과 배경 이미지의 중앙값의 차이를 구하여 이미지 콜라주가
    # 중앙에 오게 차이를 각 이미지 좌표의 y값에 빼줌.
    if len(img_map_list)!=11:
        cor = (init_index[0][1]+init_index[len(img_map_list)-2][1])/2-int(img_map_list[-1].shape[0]/2);
        cor_index = [(init_index[i][0],init_index[i][1]-int(cor)) for i in range(len(img_map_list)-1)];
    # 10개의 이미지를 모두 가져온 경우,  init_index를 그대로 가져감.
    else :
        cor_index = init_index;

    cur_index=cor_index.copy();
    # 현재 상태를 시작 좌표로 지정하여 value에 시작 상태의 점수 저장./ 평가 점수 구하는 함수 func_valid에는 매핑과 패딩이
    # 둘다 된 이미지 list가 들어가므로 매핑만 된 이미지 list를 padding_position()을 통해 패딩해준 뒤, 평가 점수 계산.

    # 크롤링을 통해 가져오기에 성공한 이미지가 하나인 경우, 이미지가 중앙부에 위치하는 좌표 하나 반환
    if len(img_map_list)==2:
        cur_index= [(init_index[0][0],init_index[0][1]-150)];

    # 그 외 2장 이상을 크롤링에 성공한 경우에는 1위와 10위를 제외한 이미지를 움직이는 행위를 반복
    else:
        img_map = padding_position(img_map_list, cur_index);
        value, map = func_valid(img_map);
        count = 0;
        incrise = True;

        # while 문 한 번의 실행은 현재 상태에서 이웃 상태들 중 높은 평가 점수를 가지는 상태로의 이동을 뜻함.
        while count < maxRounds  and incrise:

            # 현재 상태에서의 이웃 상태를 구해준 뒤, 제일 높은 점수와 그때의 상태를 next_value로 저장함.
            neighs_list=neighs(cur_index,walk,more);
            best_neigh, next_value = find_bestneighbor(img_map_list,neighs_list,cor_index);

            # next_value(이웃들 중 가장 높은 평가 점수)가 현재 상태의 평가 점수보다 높다면
            # 현재 상태를 이웃 중 가장 높은 평가점수를 가지는 상태점수로 이동
            if next_value > value:
                cur_index=best_neigh;
                value=next_value;

            # 현재 상태가 이웃 상태들 중 가장 높은 평가 점수를 가지는 상태보다 더 높은 평가 점수를 가진다면(혹은 같다면)
            # count가 maxRounds에 도달하기 전 while 문 종료
            else:
                incrise = False;

            # count 세주기
            count +=1;

            # trace의 True, False 지정을 통해 이동한 횟수와 현재 상태가 움직이는 상태들을 출력할 수 있음.
            if trace ==True:
                print(count , '번째 : ' , cur_index);

    return cur_index

def padding_position(img_resize_list,index_seg):
    ''' 이미지 리스트를 배경과 같은 사이즈로 0패딩을 해주며 이때, 이미지를 같이 받은 위치 좌표를 통해 지정하여 각각 패딩된 이미지 list를 반환하는 함수 \n
    input : img_resize_list / 패딩이 되지 않은 이미지 리스트 \n
    index_seg / 이미지 리스트의 각 이미지에서 패딩될 시 이미지가 위치할 중앙의 좌표 list  \n
    output :  각 좌표에 맞게 이미지가 배경 사이즈로 패딩된 리스트 \n
    '''

    # output을 담을 빈 list
    padding_image_list = [];

    for k in range(len(img_resize_list)):

        # 리스트의 제일 마지막은 배경을 넣으므로 패딩 해줄 필요 없이 input한 리스트의 배경을 바로 넣어주고 종료
        if k == len(img_resize_list) - 1:
            padding_image_list.append(img_resize_list[-1]);
            break

        # 우선 배경 사이즈의 0값을 가진 array를 만들어줌.
        # x,y는 각각 중앙좌표로 부터 가로 세로 길이의 절반을 빼주어 이미지가 시작하는 좌측 상단 좌표를 나타냄
        # x_,y_는 이미지의 가로 세로 길이
        temp = np.zeros(img_resize_list[-1].shape, dtype='uint8');
        x_ = img_resize_list[k].shape[1];  # 일반 list 와 np.ndarray x,y 인덱스 위치 차이
        y_ = img_resize_list[k].shape[0];
        x = round(index_seg[k][0] - 0.5 * x_);
        y = round(index_seg[k][1] - 0.5 * y_);

        # if 문은 padding_position()이 매핑된 이미지를 평가함수에 적용하기 위해 사용될 때를 나타냄. if 문 조건은 2차원 배열일 때가 조건으로
        # 0으로 채워진 array를 각 y와 x 위치로부터 세로길이, 가로 길이만큼을 이미지로 대체해줌.
        if len(img_resize_list[-1].shape)==2:
            temp[y:y + y_, x:x + x_] = img_resize_list[k];

        # if 문으로 매핑을 빼내고 elif는 최종 이미지를 생성될 때(즉, projection_image 함수에 사용될 때) 사용되며, 이때에는 jpg와 같이 3차원 채널에서
        # z 길이가 3일때 0으로 채워진 array에서 각 y와 x 위치로부터 세로길이, 가로 길이만큼을 이미지로 대체해줌.
        elif len(img_resize_list[-1].shape)==3 and img_resize_list[k].shape[-1] == 3:
            temp[y:y + y_, x:x + x_] = img_resize_list[k];

        # if 문으로 매핑을 빼내고 else는 최종 이미지를 생성될 때(즉, projection_image 함수에 사용될 때) 사용되며, 이때에는 png의 alpha 채널 정보를
        # 버리고 RGB 채널의 정보만 빼내서 0으로 채워진 array에서 각 y와 x 위치로부터 세로길이, 가로 길이만큼을 이미지로 대체해줌.
        else:
            temp[y:y + y_, x:x + x_] = img_resize_list[k][:,:,:-1];

        padding_image_list.append(temp);

    return padding_image_list

def face_index(img,classifier):
    ''' 이미지의 얼굴을 인식해서 그 좌표를 반환해주는 함수 \n
    input : img / 얼굴 분류를 할 이미지 array
            classifier / opencv2 모듈에서 제공하는 분류모델
    output : img에서 분류한 얼굴에 해당하는 \n
    (x[시작 x 좌표], y[시작 y좌표], w[가로 길이], h[세로 길이]) (muliscale로 여러 얼굴 감지 가능)
    '''

    # opencv2의 얼굴인식모델로부터 얼굴 인식을 하게 해주는 detectMultiScale() 사용하여 [x,y,w,h]를 list에 담아 반환
    faces = classifier.detectMultiScale(img);

    return faces

def crop_near_face(img,classifier):
    ''' 사람 이미지 중 전신, 혹은 상반신 사진 중 얼굴이 작게 나온 경우에 얼굴 사이즈가 이미지 사이즈에 차지하는 비율을
     너무 작지 않게 얼굴 기준으로 크롭해주는 함수 \n
    input : img / 함수 처리를 할 이미지 array
            classifier / 얼굴 인식 모델
    output : 처리된 img array '''

    # face_index 함수를 통해 이미지 내의 얼굴 정보를 가져옴.
    faces = face_index(img,classifier);

    # 다중인식 되는 경우, 하나의 얼굴만 선택 / (x,y,w,h) 중 넓이에 해당하는 w*h가 큰 경우만을 선택하고 나머지 버림.
    if len(faces) > 1:

        print('얼굴 인식이 한 명 초과로 감지됨.');

        st=0;
        temp_o=0;

        for f in faces:
            # temp=w*h로 얼굴 면적 나타냄
            temp=f[2]*f[3];
            if temp>st:     # faces 안의 좌표들 중 전 면적(st)보다 현재의 얼굴 면적(temp) 가 큰 경우
                temp_o=f;   # temp_o에 faces 중 w*h가 큰 정보를 저장
                st=temp;    # temp가 st보다 더 크므로 st 갱신
        # 제일 큰 경우 하나(temp_o)만 가지는 것으로 faces를 변경
        faces=np.array([temp_o]);

    else:
        pass

    # 다중 인식인 경우에는  면적이 큰 하나의 정보만 다시 저장하게 한 후,
    # 얼굴 가로 세로와 전체 이미지의 가로 세로의 비율에 따라 크롭
    if len(faces) == 1:

        # 세로 얼굴 길이와 전체 세로 길이 비율을 통해 세로 크롭
        if img.shape[0] >= faces[0][1]+3*faces[0][3]:
            img = img[:faces[0][1]+int(3*faces[0][3]) , : ];

        else:
            pass

        # 가로 얼굴 길이와 전체 가로 길이 비율을 통해 가로 크롭
        if img.shape[1] >= 3*faces[0][2]:

            # 세로의 경우에는 인물 이미지 얼굴이 상단에 위치하여 위에서부터 자르지만
            # 가로의 경우, 얼굴이 중앙에 혹은 측면에도 존재할 수 있음.
            # 가로의 이미지 길이가 얼굴 가로 길이보다 3배이상 길 때,
            # 가로 길이를 얼굴 가로 길이의 3배로 크롭.

            # if의 조건은 얼굴이 왼쪽에 치우친 경우로, 시작부터 얼굴 가로 길이의 3배까지 크롭
            if faces[0][0]-faces[0][2]<0:
                a , b = 0 , 3 * faces[0][2]

            # elif의 조건은 얼굴이 오른 쪽으로 치우친 경우로, 오른쪽 끝에서부터 얼굴 가로 길이의 3배 크롭
            elif faces[0][0]+2*faces[0][2]>img.shape[1]:
                a , b= img.shape[1] - 3 * faces[0][2] , img.shape[1];

            # else의 경우, 얼굴이 중앙에 있는 경우로 얼굴을 정중앙으로 하도록 얼굴 가로 길이의 3배 크롭
            else:
                a , b = faces[0][0] - faces[0][2] , faces[0][0] + 2 * faces[0][2];
                img = img[ : , a:b];

        # 가로 세로 얼굴과 가로 세로 길이의 비가 적정한 경우, pass
        else:
            pass

    # classifier가 얼굴 인식을 하지 못한 경우, 가로 세로 길이 비율을 따져서 세로가 긴 경우만 반으로 크롭
    else:
        print('얼굴 인식이 되지 않음');

        # 이미지 세로 길이 >  2 * 이미지 가로 길이인 경우, 세로를 위에서부터 전체 세로 길이의 절반으로 크롭
        if img.shape[0] > img.shape[1] * 2:
            img = img[:int(img.shape[0] / 2), :]

        else:
            pass

    return img

def projection_image(img_padding_list):

    ''' 순위가 높은 이미지가 제일 위, 배경이 제일 아래라 생각하고 쌓인 이미지를
    위에서 아래로 투영한다고 생각하고 짠 함수 \n
    input : img_padding_list / 배경 사이즈로 패딩이 끝난 이미지들의 list \n
    output : 각 이미지가 콜라주되어 보여질 최종 이미지를 나타내는 array'''

    # 최종으로 보여질 이미지를 저장할 array 사이즈의 zeros
    final_image = np.zeros(img_padding_list[-1].shape, dtype='uint8');
    # 각 위치에서 list의 몇번째 이미지가 저장되었는 지 알 수 있는 2차원 array
    final_record = np.zeros((img_padding_list[-1].shape[0], img_padding_list[-1].shape[0]), dtype='uint8');

    for x_index in range(img_padding_list[-1].shape[1]):
        for y_index in range(img_padding_list[-1].shape[0]):
            # 각 x, y 좌표를 하나씩 옮겨가면서
            for z_index in range(len(img_padding_list)):
                # z는 list 내의 index를 의미하며 각 픽셀에 대하여 list의 앞에서부터
                # if는 픽셀 위치에서 RGB의 합이 0이라면 pass
                if sum(img_padding_list[z_index][y_index, x_index]) == 0:
                    continue
                # else는 픽셀 위치에서 RGB의 합이 0이 아니라면 그 정보를 최종 이미지의 픽셀 자리에 저장
                # record에는 index 넘버를 저장
                else:
                    final_image[y_index, x_index] = img_padding_list[z_index][y_index, x_index];
                    final_record[y_index, x_index] = z_index;
                    break

    return final_image, final_record

def make_classifier(xml_index):
    ''' opencv에서 제공하는 얼굴인식 모델 생성 함수 \n
    input : xml_index / opencv에서 제공하는 얼굴 인식 모델 xml로 깃헙에서 다운 가능 \n
    output : 얼굴 분류 모델 '''
    classifier = cv2.CascadeClassifier(xml_index);

    return classifier

def web_index(img_resize_list,index):
    '''앞에서 작성한 함수들 중 인덱스 list를 반환하는 함수들이 각 이미지의 중앙 위치를 반환하여
    이를 다시 좌측 상단인 이미지 시작 인덱스를 나타내도록 변환해주는 함수
    input : img_list / resize된 패딩되기 전의 이미지를 모아놓은 list로, 이미지의 가로 세로 길이를 가져오기 위해 받음.
            index / 각 이미지가 배경 사이즈에서 위치하게 될 중앙 좌표를 모아놓은 list
    output : 각 이미지가 배경 사이즈에서 위치하게 될 시작 위치인 좌측 상단 좌표를 모아놓은 list'''

    b_index = [];
    for i in range(len(img_resize_list)):

        # 좌표 = 중앙 x 좌표 - 이미지 가로 길이 절반 , 중앙 y 좌표 - 이미지 세로 길이 절반
        temp = index[i][0]-int(img_resize_list[i].shape[1]/2) , index[i][1]-int(img_resize_list[i].shape[0]/2);
        b_index.append(temp);

    return b_index

def shadow(img,dist=[5,5],tu=0.5):
    '''  png 이미지의 알파 채널 정보를 응용하여 이미지의 이미지를 생성해주는 함수 \n
    input : img / 그림자를 넣을 이미지 array 한장
            dist/ 그림자와 이미지간 거리로, [우측 거리, 아래 방향 거리] (0보다 큰 정수값 입력) \n
            tu / 투명도 조절을 위한 변수로, 0에 가까울 수록 투명해지며 1에 가까울 수록 불투명해짐.\n
    output : input한 이미지의 그림자가 생긴 이미지
    '''

    # 그림자만 넣을 빈 array 생성
    img_copy=np.zeros(img.shape);

    #빈 array에 값 넣어주기
    for i in range(img.shape[-1]):

        # alpha 채널에 정보넣기
        if i ==3:
            img_copy[:,:,i]=img[:,:,-1]*tu;

        # 그 외의 채널엔 검정 RGB(0,0,0)이므로 0으로 설정
        else:
            img_copy[:,:,i]=0;


    # float 유형 uint8을 변경하면서 0~255 사이 정수값으로 변경
    img_copy.astype('uint8');
    # return 값 받을 빈 array 생성
    img_plus=np.zeros((img.shape[0]+dist[1],img.shape[1]+dist[0],img.shape[2]));
    # 그림자 먼저 덮기
    img_plus[dist[1]:,dist[0]:,:]=img_copy;

    # 원래 이미지 넣기/ 그림자보다 이미지가 앞에 있으니, 이미지의 투명도값이 0이 아니라면 이미지 정보 넣기
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if img[y,x,-1]!=0:
                img_plus[y,x]=img[y,x];

    # 이미지 표현을 위해 0~255 사이 정수값으로 정렬
    img_plus.astype('uint8');

    # 하얀 이미지의 경우, 흰 배경에서 안보이는 것을 방지하기 위해, RGB 값 살짝 낮추기
    img_plus = np.where(img_plus == [255, 255, 255, 255], [240, 240, 240, 255], img_plus);

    return img_plus