### This is package for model serving 

docker run -it -p 80:8080 -e JWT_AUTH_SECRET_KEY=${JWT_AUTH_SECRET_KEY} -e JWT_AUTH_ALGORITHM=${JWT_AUTH_ALGORITHM} -e DB_HOST={DB_HOST} -e DB_NAME=${DB_NAME} -e DB_USER=${DB_USER} -e DB_PASS=${DB_PASS} ${containerRegistry}/${imageRepository}:${tag}