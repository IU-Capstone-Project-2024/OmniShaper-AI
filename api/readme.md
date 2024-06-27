# API documentation
## Start up
To start in dev mode \
Run `fastapi dev main.py`

In the terminal you will see a big yellow box, saying on which localhost it will run. Please, use `http://127.0.0.1:8000/docs` (standard URL unless other port is set) for testing.
## Functionality
`create_request` will ask the user for a prompt. It will generate the `file_id` for future files to download and will initiate `promt_to_3D` function in `ml_pipline.final_model` package. If something will throw an exception during the process, it will instead throw an HTTP exception with status code 500 which will state that the algorithm did not finish the execution.

There are 3 types of different functions that have the same meaning behind them - to download a file:
* `download_obj`
* `download_mtl`
* `download_png`
  
All of them will ask the user to provide the `file_id`, please, copy it when executing `create_request` and put them in the `download` type command to get your file.
In case you entered the wrong `file_id`, server will issue an HTTP 404 exception, stating, that the file is not found.

PLEASE, DO NOT USE `download_obj` WHILE TESTING. Swagger UI does not support prewieving 3D objects, hence it will try to show you raw file that has over 60k lines of data in it. It will eat all of your local computer's resources. Be aware. 
