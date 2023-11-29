import os
import shutil

def rename_files(source_directory):
    # 새파일 디렉토리
    target_directory = os.path.join(source_directory, 'renamed_txt')
    os.makedirs(target_directory, exist_ok=True)

    # .txt 불러오기
    file_paths = [f for f in os.listdir(source_directory) if f.endswith('.txt')]

    # 이름 변경 후 이동
    for file_name in file_paths:
        # 새이름
        new_file_name = file_name.split('_jpeg')[0] + '.txt'
        new_file_path = os.path.join(target_directory, new_file_name)

        # 파일 복사
        shutil.copy(os.path.join(source_directory, file_name), new_file_path)
        print(f"Renamed and moved '{file_name}' to '{new_file_path}'")

# Example Usage
source_directory = '/Users/sdc/Documents/ds_study/DeepLearning/dl_project/prototype_v1/name_change_test/label'  # 라벨 디렉토리
rename_files(source_directory)
