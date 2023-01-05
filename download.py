import requests
import os
import conf as cfg

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    # URL = "https://drive.google.com/file/d/1c84_CtRGK8ifv-7c9MI_xYVruYAnYX5p/view?usp=share_link"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id , 'confirm': 1 }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    # file_id = '1c84_CtRGK8ifv-7c9MI_xYVruYAnYX5p'
    file_id = '1zG5UwkPVCrTKNKDqPUcMiGLvuvq7GHd9'
    destination = os.path.join(cfg.paths['data'], 'dataset.zip')
    download_file_from_google_drive(file_id, destination)





# def download_file_from_google_drive(id, destination):
#     URL = "https://docs.google.com/uc?export=download"
#     # URL = "https://drive.google.com/file/d/1c84_CtRGK8ifv-7c9MI_xYVruYAnYX5p/view?usp=share_link"

#     session = requests.Session()

#     response = session.get(URL, params = { 'id' : id }, stream = True)
#     token = get_confirm_token(response)

#     if token:
#         params = { 'id' : id, 'confirm' : token }
#         response = session.get(URL, params = params, stream = True)

#     save_response_content(response, destination)    

# def get_confirm_token(response):
#     for key, value in response.cookies.items():
#         if key.startswith('download_warning'):
#             return value

#     return None

# def save_response_content(response, destination):
#     CHUNK_SIZE = 32768

#     with open(destination, "wb") as f:
#         for chunk in response.iter_content(CHUNK_SIZE):
#             if chunk: # filter out keep-alive new chunks
#                 f.write(chunk)

# if __name__ == "__main__":
#     file_id = '1c84_CtRGK8ifv-7c9MI_xYVruYAnYX5p'
#     destination = cfg.paths['data']
#     download_file_from_google_drive(file_id, destination)