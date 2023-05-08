import cv2
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 512

def crop_blank(img):
    '''
    세그멘테이션 이후에 이미지의 빈공간을 제거하는 함수

    img : np.array / 어레이형태의 이미지
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY);
    index = cv2.boundingRect(gray);
    (x,y,w,h)=index;
    img_crop = img[y:y + h + 1, x : x + w + 1, :];

    return img_crop

def makeWhiteEdge(img_arr, k_size=3, iterations=3):
    '''
    이미지간의 구분을 위하여 이미지의 주변에 흰색의 테두리를 만들어 내는 함수

    img_arr : np.array / 어레이형태의 이미지 
    k_size : int / 테두리 생성에 필요한 이미지 팽창 연산에 사용되는 커널의 크기
    iterations : int / 팽창연산을 얼마나 반복할 것인지 지정
    '''

    # 이미지 팽창을 적용하기에 앞서 제로패딩을 실시
    pad_num = 50
    paddings = tf.constant([[pad_num, pad_num], [pad_num, pad_num]])
    try:
        r,g,b,alpha = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2], img_arr[:,:,3]        
        r,g,b,alpha = (np.array(tf.pad(r, paddings, "CONSTANT")), 
                       np.array(tf.pad(g, paddings, "CONSTANT")), 
                       np.array(tf.pad(b, paddings, "CONSTANT")),
                       np.array(tf.pad(alpha, paddings, "CONSTANT")) )
        img_arr = np.stack([r,g,b,alpha], axis=2)

        # 이미지의 팽창을 적용한 알파채널을 추출
        im_alpha = img_arr[:,:,3]
        # 알파채널에 팽창을 적용
        kernel = np.ones((k_size,k_size), np.uint8)
        result = cv2.dilate(im_alpha, kernel, iterations=iterations)
        # 팽창이후에 팽창결과와 원 알파채널을 마이너스 연산하여 흰색테두리가 들어갈 위치를 정의
        white_space_mask = np.stack([np.where(result - im_alpha>0, True,False)]*3, axis=2)
        # 알파채널은 팽창한 결과를 가져가고
        img_arr[:,:,3] = result
        # rgb채널은 각각 255를 가지게하여 흰색으로 칠함
        img_arr[:,:,:3][white_space_mask] = 255
        # 빈공간을 크롭하고 리턴
        return crop_blank(img_arr)
    # 이미지가 4차원 alpha 채널을 가지지 않으면 원본에서 빈공간을 크롭하고 리턴
    except:
        return crop_blank(img_arr)
    

def read_image(image_path, original=False):
    '''
    이미지를 tensorflow 턴서 형태로 읽어오는 함수

    image_path : str / 이미지의 경로
    original : bool / 원본형태를 유지할 것인지 세그멘테이션 모델에 들어갈 수 있는 크기로 정규화 및 리사이즈 할 것인지
    '''
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image.set_shape([None, None, 3])
    if not original:
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image/255
    return image

def segSave(model, image_path, save_path=None):
    '''
    이미지를 모델에 넣어 세그멘테이션 완료한 이미지를 얻고 원본 사이즈로 복원하는 함수
    이미지사이즈 복원이후에 한객체에 생긴 구멍으로 보이는 물체(악세사리로 추정되는)는 채워리턴

    model : tf.model / 기학습한 세그멘테이션 모델 
    image_path : 인풋으로 들어갈 원본이미지의 경로
    save_path : 아웃풋으로 저장한 이미지의 경로

    '''
    original_image = read_image(image_path, original=True)
    input_image = read_image(image_path)
    # 모델로 예측을 실행
    input_pred = model.predict(input_image[np.newaxis,...])
    # 사람에 해당하는 픽셀이라고 예측할 기준
    thresh_hold = 0.4
    # 기준에 따라 마스크 생성
    pred_mask = input_pred > thresh_hold
    # 마스크를 원본이미지의 사이즈로 복원
    alpha = cv2.resize(np.squeeze(np.where(pred_mask, 1.0,0)), (original_image.shape[1], original_image.shape[0]))
    alpha = np.where(alpha>0,1,0).astype(np.float32)
    # 마스크에 구멍이 있을 경우
    try:
        # 다리사이 같은 부분을 컨투어로 인식하는 것을 방지하기위해 제로패딩
        pad_num = 50
        paddings = tf.constant([[pad_num, pad_num], [pad_num, pad_num]])
        padded = np.array(tf.pad(alpha, paddings, "CONSTANT"))
        # 마스크의 구멍을 확인하고 채움
        contours_coor ,info = cv2.findContours(padded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        info_ = np.where(info[0][:,-1]==-1)[0]
        for i, num in enumerate(info_):
            contour_coor = contours_coor[num]
            if i == 0:
                filled_hole = cv2.fillPoly(padded.astype(np.uint8), [np.squeeze(contour_coor)], 255)
            else :
                filled_hole = cv2.fillPoly(filled_hole, [np.squeeze(contour_coor)], 255)
        filled_hole_ = filled_hole[pad_num:padded.shape[0]-pad_num, pad_num:padded.shape[1]-pad_num]
        filed_mask = np.where(filled_hole_[...,np.newaxis] > 0, 1,0)
        masked_image = original_image * filed_mask
        masked_image_png = np.concatenate([masked_image[:,:,2][...,np.newaxis], masked_image[:,:,1][...,np.newaxis],
                                        masked_image[:,:,0][...,np.newaxis] ,(filed_mask).astype(np.uint8)*255], axis=2)
    # 마스크에 구멍이 없을 경우
    except:
        alpha_ = alpha[...,np.newaxis]
        masked_image = original_image * alpha_
        masked_image_png = np.concatenate([masked_image[:,:,2][...,np.newaxis], masked_image[:,:,1][...,np.newaxis],
                                        masked_image[:,:,0][...,np.newaxis], (alpha_).astype(np.uint8)*255 ], axis=2)
    # 저장
    if save_path:
        cv2.imwrite(save_path, masked_image_png)

    return masked_image_png

# def segSave(model, image_path, save_path=None):
#     original_image = read_image(image_path, original=True)
#     input_image = read_image(image_path)
#     input_pred = model.predict(input_image[np.newaxis,...])

#     thresh_hold = 0.3
#     pred_mask = input_pred > thresh_hold
#     alpha = cv2.resize(np.squeeze(np.where(pred_mask, 1.0,0)), (original_image.shape[1], original_image.shape[0]))
#     alpha = np.where(alpha>0,1,0).astype(np.float32)

#     try:
#         contours_coor ,info = cv2.findContours(alpha.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#         info_ = np.where(info[0][:,-1]==-1)[0]
#         for i, num in enumerate(info_):
#             contour_coor = contours_coor[num]
#             if i == 0:
#                 filled_hole = cv2.fillConvexPoly(alpha.astype(np.uint8), np.squeeze(contour_coor), 255)
#             else :
#                 filled_hole = cv2.fillConvexPoly(filled_hole, np.squeeze(contour_coor), 255)
#         filed_mask = np.where(filled_hole[...,np.newaxis] > 0, 1,0)
#         masked_image = original_image * filed_mask
#         masked_image_png = np.concatenate([masked_image[:,:,2][...,np.newaxis], masked_image[:,:,1][...,np.newaxis],
#                           masked_image[:,:,0][...,np.newaxis] ,(filed_mask).astype(np.uint8)*255], axis=2)
#     except:
#         alpha_ = alpha[...,np.newaxis]
#         masked_image = original_image * alpha_
#         masked_image_png = np.concatenate([masked_image[:,:,2][...,np.newaxis], masked_image[:,:,1][...,np.newaxis],
#                                            masked_image[:,:,0][...,np.newaxis], (alpha_).astype(np.uint8)*255 ], axis=2)
#     if save_path:
#         cv2.imwrite(save_path, masked_image_png)

#     return masked_image_png