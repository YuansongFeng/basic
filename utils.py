import os
import cv2

# output a list of images to the output_dir
def output_list_imgs(list_imgs, output_dir='output'):
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    for idx, img in enumerate(list_imgs):
        cv2.imwrite(os.path.join(output_dir, 'image_%i.jpg' % idx), img)
        print('image output to ' + os.path.join(output_dir, 'image_%i.jpg' % idx))

# output a list of list of images to output_dir
def output_list_list_imgs(list_list_imgs, output_dir='output'):
    for idx, list_imgs in enumerate(list_list_imgs):
        curr_output_dir = os.path.join(output_dir, str(idx))
        output_imgs(list_imgs, curr_output_dir)
