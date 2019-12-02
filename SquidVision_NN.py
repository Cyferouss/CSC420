import numpy as np
import matplotlib as mpl
import pandas as pd
import random
import cv2

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, GlobalAveragePooling2D, Conv2D,MaxPool2D, ZeroPadding2D, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as kr
from sklearn.model_selection import train_test_split

def plot_results(images, coordinates, image_size=96):
    coordinate_sets = []
    for point_set in coordinates:
        x = []
        y = []
        for i in range(len(point_set)):
            if i % 2 == 0 :
                x.append(point_set[i] * image_size)
            else:
                y.append(point_set[i] * image_size)
        
        coordinate_sets.append((x, y))

    fig, ax = mpl.pyplot.subplots(nrows = 4, ncols = 4, sharex=True, sharey=True, figsize = (16,16))
    
    for row in range(4):
        for col in range(4):
            index = random.randint(0, len(images) - 1)
            image = np.reshape(images[index], (96,96))
            landmark_x, landmark_y = coordinate_sets[index][0], coordinate_sets[index][1]
            print(np.shape(image))
            ax[row, col].imshow(image, cmap="gray")
            ax[row, col].scatter(landmark_x, landmark_y, c = 'r')
            ax[row, col].set_xticks(())
            ax[row, col].set_yticks(())
            ax[row, col].set_title('Index Number: %d' %index)
            
    mpl.pyplot.show()

def load_dataset(set_size=16, folder_path='test_images/', extension_name='.png'):
    image_set = []
    for i in range(set_size):
        file_path = folder_path + str(i) + extension_name
        print(file_path)
        
        img = np.float32(cv2.imread(file_path))
        g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rr_img = np.expand_dims(cv2.resize(g_img, (96,96), interpolation=cv2.INTER_CUBIC), axis=-1)
        image_set.append(rr_img)
        
    return np.array(image_set)

def sourced_cnn():
    #sourced from paper that we referred to
    model = Sequential()
    
    model.add(Convolution2D(64, (3,3), padding='valid', use_bias=False, input_shape=(96,96,1)))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(BatchNormalization())
    
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, (2,2), padding='valid', use_bias=False))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(BatchNormalization())
    
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(256, (2,2), padding='valid', use_bias=False))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(BatchNormalization())
    
    model.add(MaxPool2D(pool_size=(2, 2)))
     
    model.add(Flatten())
    model.add(Dense(500,activation='relu', kernel_regularizer=regularizers.l2(0.003)))
    model.add(Dense(500,activation='relu', kernel_regularizer=regularizers.l2(0.001)))

    model.add(Dense(8, activation='sigmoid'))
    
    model.compile(optimizer='SGD', 
              loss='mean_squared_error',
              metrics=['mae', 'accuracy'])
    
    return model

def homographical_augment(image_set, truth_set):
    augmented_images = []
    augmented_truths = []
    flipped_images = []
    flipped_truths = []
    
    #unpack coordinate sets for truth_set
    coordinate_sets = []
    for point_set in truth_set:
        x = []
        y = []
        for i in range(len(point_set)):
            if i % 2 == 0 :
                x.append(np.uint32(point_set[i] * 96))
            else:
                y.append(np.uint32(point_set[i] * 96))
        
        coordinate_sets.append((x, y))
        
    #Create Homography_matrices
    h_matrices = []
    
    pts_src_h1 = np.array([[0,0], [0, 96], [96, 0], [96,96]])
    pts_dst_h1 = np.array([[0,0], [0, 96], [80, 16], [80, 80]])
    h1, status = cv2.findHomography(pts_src_h1, pts_dst_h1)
    h_matrices.append(h1)
    
    pts_src_h2 = np.array([[0,0], [0, 96], [96, 0], [96,96]])
    pts_dst_h2 = np.array([[0,0], [0, 96], [72, 24], [72, 72]])
    h2, status = cv2.findHomography(pts_src_h2, pts_dst_h2)
    h_matrices.append(h2)
    
    pts_src_h3 = np.array([[0,0], [0, 96], [96, 0], [96,96]])
    pts_dst_h3 = np.array([[0,0], [0, 96], [66, 30], [66, 66]])
    h3, status = cv2.findHomography(pts_src_h3, pts_dst_h3)
    h_matrices.append(h3)
    
    pts_src_h4 = np.array([[0,0], [0, 96], [96, 0], [96,96]])
    pts_dst_h4 = np.array([[0,0], [0, 96], [60, 16], [60, 80]])
    h4, status = cv2.findHomography(pts_src_h4, pts_dst_h4)
    h_matrices.append(h4)
    
    pts_src_h5 = np.array([[0,0], [0, 96], [96, 0], [96,96]])
    pts_dst_h5 = np.array([[16,16], [16, 80], [96, 0], [96, 96]])
    h5, status = cv2.findHomography(pts_src_h5, pts_dst_h5)
    h_matrices.append(h5)
    
    pts_src_h6 = np.array([[0,0], [0, 96], [96, 0], [96,96]])
    pts_dst_h6 = np.array([[24,24], [24, 72], [96, 0], [96, 96]])
    h6, status = cv2.findHomography(pts_src_h6, pts_dst_h6)
    h_matrices.append(h6)

    pts_src_h7 = np.array([[0,0], [0, 96], [96, 0], [96,96]])
    pts_dst_h7 = np.array([[30,30], [30, 66], [96, 0], [96, 96]])
    h7, status = cv2.findHomography(pts_src_h7, pts_dst_h7)
    h_matrices.append(h7)
    
    pts_src_h8 = np.array([[0,0], [0, 96], [96, 0], [96,96]])
    pts_dst_h8 = np.array([[36,16], [36, 80], [96, 0], [96, 96]])
    h8, status = cv2.findHomography(pts_src_h8, pts_dst_h8)
    h_matrices.append(h8)
    
    augmented_coordinates = []
    flipped_coordinates = []
    for i in range(len(image_set)):
        #Calculate Homography Matrix
        h = random.choice(h_matrices)
        #Augment Image
        reshaped_img = np.reshape(image_set[i], (96,96))
        augmented_image = cv2.warpPerspective(reshaped_img, h, (96,96))
        
        #Create Truth Matrix and populate
        t_matrix = np.zeros(np.shape(image_set[i]))
        coordinate_set = coordinate_sets[i]
        for j in range(len(coordinate_set[0])):
            x_pos = coordinate_set[0][j]
            y_pos = coordinate_set[1][j]
            t_matrix[y_pos][x_pos] = 1
            
        augmented_t_matrix = cv2.warpPerspective(t_matrix, h, (96,96))        
        #Local Non-maxima suppression
        for p in range(96):
            for k in range(96):
                value = augmented_t_matrix[p][k]
                if(value != 0 and p > 2 and k > 2):
                    matrix_slice = augmented_t_matrix[p-2:p+2, k-2:k+2]
                    lnms_matrix = local_non_max_suppression(matrix_slice)
                    augmented_t_matrix[p-2:p+2, k-2:k+2] = lnms_matrix
                    
        #Create Truths for Homography
        augmented_x = []
        augmented_y = []
        for p in range(96):
            for k in range(96):
                value = augmented_t_matrix[p][k]
                if(value != 0):       
                    augmented_x.append(k / 96)
                    augmented_y.append(p / 96)
        
        if(len(augmented_x) == 4):
            augmented_coordinates.append((augmented_x, augmented_y))
            augmented_images.append(np.expand_dims(augmented_image, axis=-1))

        #Flip augmentation

        flip = random.choice([0,1,-1])
        flipped_image = cv2.flip(reshaped_img, flip)
        flipped_t_matrix = cv2.flip(t_matrix, flip)
        flipped_x = []
        flipped_y = []
        for u in range(96):
            for v in range(96):
                value = flipped_t_matrix[u][v]
                if(value != 0):       
                    flipped_x.append(v / 96)
                    flipped_y.append(u / 96)
        
        if(len(flipped_x) == 4):
            flipped_coordinates.append((flipped_x, flipped_y))
            flipped_images.append(np.expand_dims(flipped_image, axis=-1))
            
#    Repack coordinates into truth set
    augmented_truths = repack_truths(augmented_coordinates)
    flipped_truths = repack_truths(flipped_coordinates)
    
    augmented_img_set = np.concatenate((augmented_images, flipped_images), axis=0)
    augmented_truth_set = np.concatenate((augmented_truths, flipped_truths), axis=0)
    
    return augmented_img_set, augmented_truth_set

def local_non_max_suppression(matrix):
    lnms_matrix = matrix.copy()
    x_max, y_max, curr_max = 0, 0, -1
    
    for y in range(np.size(lnms_matrix, 0)):
        for x in range(np.size(lnms_matrix, 1)):
            if lnms_matrix[y][x] > curr_max:
                if curr_max > -1:
                    lnms_matrix[y_max][x_max] = 0
                curr_max = lnms_matrix[y][x]
                x_max = x
                y_max = y
            else:
                lnms_matrix[y][x] = 0
                
    lnms_matrix[y_max][x_max] = 1
    
    return lnms_matrix

def repack_truths(coordinates_set):
    truth_set = []
    for coordinate_set in coordinates_set:
        truth = np.zeros((8,))
        truth[0] = coordinate_set[0][0]
        truth[1] = coordinate_set[1][0]
        truth[2] = coordinate_set[0][1]
        truth[3] = coordinate_set[1][1]
        truth[4] = coordinate_set[0][2]
        truth[5] = coordinate_set[1][2]
        truth[6] = coordinate_set[0][3]
        truth[7] = coordinate_set[1][3]
        truth_set.append(truth)
        
    return truth_set

def process_dataset(face_images, d_pts):
    non_zero_selection = np.nonzero(d_pts.left_eye_center_x.notna() & 
                           d_pts.right_eye_center_x.notna() & 
                           d_pts.nose_tip_x.notna() & 
                           d_pts.mouth_center_bottom_lip_x.notna())[0]

    #image_size 96x96
    image_size = face_images.shape[1]
    set_size = non_zero_selection.shape[0]

    #Creating dataset and truthsets    
    data_set = np.zeros((set_size, image_size, image_size, 1))
    truth_set = np.zeros((set_size, 8))

    true_index = 0    
    for index in non_zero_selection:  
        #Upscale 96x96 image to 192x192
        resized_img = cv2.resize(face_images[index], (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        data_set[true_index] = np.expand_dims(resized_img, axis=-1)
        
        # One-hot encoded pixel vectors scaled to 0-1, after rescaling to 192x192
        truth_set[true_index][0] = (d_pts.left_eye_center_x[index]) / image_size
        truth_set[true_index][1] = (d_pts.left_eye_center_y[index]) / image_size
        truth_set[true_index][2] = (d_pts.right_eye_center_x[index])  / image_size
        truth_set[true_index][3] = (d_pts.right_eye_center_y[index]) / image_size
        truth_set[true_index][4] = (d_pts.nose_tip_x[index]) / image_size
        truth_set[true_index][5] = (d_pts.nose_tip_y[index]) / image_size
        truth_set[true_index][6] = (d_pts.mouth_center_bottom_lip_x[index]) / image_size
        truth_set[true_index][7] = (d_pts.mouth_center_bottom_lip_y[index]) / image_size
        
        true_index += 1
        
    return data_set, truth_set

def upscale_coordinates(coordinate_set, starting_size, upscaled_size):
    coordinates = []
    x_factor = upscaled_size[0] / starting_size[0]
    y_factor = upscaled_size[1] / starting_size[1]
    
    for i in range(len(coordinate_set)):
        if i % 2 == 0:
            coordinates.append(coordinate_set[i] * x_factor)
        else:
            coordinates.append(coordinate_set[i] * y_factor)
        
    return coordinates

def downscale_coordinates(coordinate_set, starting_size, downscaled_size):
    coordinates = []
    x_factor = downscaled_size[0] / starting_size[0]
    y_factor = downscaled_size[1] / starting_size[1]
    
    for i in range(len(coordinate_set)):
        if i % 2 == 0:
            coordinates.append(coordinate_set[i] * x_factor)
        else:
            coordinates.append(coordinate_set[i] * y_factor)
        
    return coordinates

def draw_nose(image, coordinate_set, rgb=False):
    raw_nose = cv2.imread('SquidNose.png')
    gray_nose = cv2.cvtColor(raw_nose, cv2.COLOR_BGR2GRAY)
    
    #nose_pts = np.array(np.float32([[0,0], [568,0], [287,313], [275,681]]))
    nose_pts = np.array(np.float32([[0,0], [568,0], [0,681], [568,681]]))

    #Coordinates[0] = Left Eye
    #Coordinates[1] = Right Eye
    #Coordinates[2] = Nose
    #Coordinates[3] = Mouth
    
    coordinates = []
    coordinate = []
    for i in range(len(coordinate_set)):
        if i % 2 == 0:
            coordinate.append(coordinate_set[i] * 96)
        else:
            coordinate.append(coordinate_set[i] * 96)
            coordinates.append(coordinate.copy())
            coordinate = []

    coordinates = np.array(np.float32(coordinates))
        #Tl, TR, BL, BR
    nose_box_lx = coordinates[0][0]
    nose_box_rx = coordinates[1][0]
    nose_box_ty = coordinates[0][1]
    nose_box_by = coordinates[3][1]
    
    #Rectangle BB around Nose
    
    bb_coordinates = np.array([
            [nose_box_lx, nose_box_ty],
            [nose_box_rx, nose_box_ty],
            [nose_box_lx, nose_box_by],
            [nose_box_rx, nose_box_by]
            ])
    
    print(coordinates)
    print(bb_coordinates)
    print(nose_pts)
    
    h, res = cv2.findHomography(nose_pts, bb_coordinates)
    print(h)
    
    if rgb == False:
        warped_nose = cv2.warpPerspective(gray_nose, h, (np.size(image,0), np.size(image,1)))
        print(np.shape(image))
        print(np.shape(warped_nose))
        augmented_image = image.copy()
        
        for y in range(np.size(image, 0) - 15):
            for x in range(np.size(image,1) - 15):
                nose_pixel = warped_nose[y][x]
                if nose_pixel != 0 and nose_pixel != 255:
                    augmented_image[y][x] = nose_pixel
    else:
        nose_b, nose_g, nose_r = cv2.split(raw_nose)
        img_b, img_g, img_r = cv2.split(image)
        
        warped_nose_b = cv2.warpPerspective(nose_b, h, (np.size(image,0), np.size(image,1)))
        warped_nose_g = cv2.warpPerspective(nose_g, h, (np.size(image,0), np.size(image,1)))
        warped_nose_r = cv2.warpPerspective(nose_r, h, (np.size(image,0), np.size(image,1)))
        
        print(np.shape(warped_nose_b))
        print(np.shape(warped_nose_g))
        print(np.shape(warped_nose_r))
        
        print(np.shape(img_b))
        print(np.shape(img_g))
        print(np.shape(img_r))
        for y in range(np.size(image, 0)):
            for x in range(np.size(image,1)):
                nose_pixel_b = warped_nose_b[y][x]
                nose_pixel_g = warped_nose_g[y][x]
                nose_pixel_r = warped_nose_r[y][x]
                if nose_pixel_b > 20 and nose_pixel_b < 245:
                    img_b[y][x] = nose_pixel_b
                    img_g[y][x] = nose_pixel_g
                    img_r[y][x] = nose_pixel_r
                    
        augmented_image = cv2.merge((img_b, img_g, img_r))
        
    return augmented_image
if __name__ == "__main__":
#    Built with the help of https://www.kaggle.com/soumikrakshit/eye-detection-using-facial-landmks
#    Load Dataset and CSV
#    face_images = np.moveaxis(np.load('face_images.npz')['face_images'],-1,0)
#    d_pts = pd.read_csv('facial_keypoints.csv')
#    
#    #Process Data_set
#    data_set, truth_set = process_dataset(face_images, d_pts)
#    
#    #Get augmented data
#    a_t, a_tr = homographical_augment(data_set, truth_set)
#
#    #Append augmented data
#    x_set = np.concatenate((data_set, a_t), axis=0)
#    y_set = np.concatenate((truth_set, a_tr), axis=0)
#    
#    #Split into training and test sets
#    x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.2, random_state=10)
#    
#    plot_results(x_train, y_train)
    
    #split between training set and validation set    
#    model = sourced_cnn()
#    model.load_weights('ghetto_net_weights_use.h5')
#    checkpoint = ModelCheckpoint('ghetto_net_weightsv4.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=1)
#    model.fit(x_train, y_train, batch_size=16, epochs=100, callbacks=[checkpoint], validation_split=0.17, verbose=1, shuffle=True)
     
#    predictions = model.predict(x_test)
    
    #Real portrait testing
    testing_data = load_dataset()
    predicted_testing = model.predict(testing_data)
    #plot_results(x_test, predictions)
    
    
    raw_image = cv2.imread('test_images/15.png')
    gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    
    print(predicted_testing[7])
    img_upscaled_dpts = upscale_coordinates(predicted_testing[15], (96,96), (np.size(raw_image,0),np.size(raw_image,1)))
    print(predicted_testing[7])
    print(img_upscaled_dpts)
    
#    image = np.reshape(x_test[1762], (96,96))
#    coordinates = predictions[1762]
    nose = draw_nose(raw_image, img_upscaled_dpts, rgb=True)
    
    nose_rgb = cv2.cvtColor(nose, cv2.COLOR_BGR2RGB)
    mpl.pyplot.imshow(nose_rgb)
    
        
    