FROM python:3.10.9

# set the directory
WORKDIR C:\Users\bouab\DEV\Towards_Data_Science\NLP\playground\Language_Model_with_FNN
# copy all the files in the directory to the docker image
COPY . .

# install the requirements
RUN pip install --no-cache-dir -r requirements_copy_linux.txt

# expose some port
EXPOSE 4567

# run the command
CMD ["python", "FFNLM.py"]

