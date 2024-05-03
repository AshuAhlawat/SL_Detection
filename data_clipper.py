import os
import shutil

new_modified_dir = 'modified_dir'
if not os.path.exists(new_modified_dir):
    os.mkdir(new_modified_dir)

base_dir = r'asl_train\asl_train'
for subfolder in os.listdir(base_dir):

    target_subfolder_path = os.path.join(new_modified_dir, subfolder)
    if not os.path.exists(target_subfolder_path):
        os.mkdir(target_subfolder_path)

    i = 0
    
    current_subfolder_path = os.path.join(base_dir, subfolder)
    
    for image in os.listdir(current_subfolder_path):

        if i >= 4000:
            break
        
        src_path = os.path.join(current_subfolder_path, image)
        dst_path = os.path.join(target_subfolder_path, image)
        
        shutil.move(src_path, dst_path)
        i += 1


