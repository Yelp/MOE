FROM ubuntu:14.04
MAINTAINER Scott Clark <sclark@yelp.com> and Eric Liu <eliu@yelp.com>

# Configure a non-privileged user named app. It is highly suggested you do not
# run your application as root.
RUN addgroup --gid 9999 app &&\
    adduser --uid 9999 --gid 9999 --disabled-password --gecos "Application" app &&\
    usermod -L app

# Install software from Ubuntu.
RUN apt-get update
RUN apt-get install -y build-essential gcc python python-dev python2.7 python2.7-dev cmake libboost-all-dev doxygen libblas-dev liblapack-dev gfortran git make flex bison libssl-dev libedit-dev

# Install pip systemwide for Python.
ADD https://raw.github.com/pypa/pip/master/contrib/get-pip.py /tmp/get-pip.py
RUN python /tmp/get-pip.py

# Install requirements (done here to allow for caching)
ADD requirements.txt /home/app/MOE/
RUN cd /home/app/MOE/ && pip install -r requirements.txt

# Copy over the code
ADD . /home/app/MOE/
WORKDIR /home/app/MOE

# Install the python
ENV MOE_NO_BUILD_CPP True
RUN pip install -e . && python setup.py install

# Build the C++
WORKDIR /home/app/MOE/moe
RUN mkdir build
WORKDIR /home/app/MOE/moe/build
RUN cmake /home/app/MOE/moe/optimal_learning/cpp/
RUN make

# Copy the built C++ into the python
RUN cp -r /home/app/MOE/moe/build /usr/local/lib/python2.7/dist-packages/moe/.

RUN chown -R app:app /home/app/MOE && chmod -R a+r /home/app/MOE
WORKDIR /home/app/MOE

# Run tests
RUN testify -v moe.tests

# Configure docker container.
EXPOSE 6543

# Set up the webserver
USER app
CMD ["development.ini"]
ENTRYPOINT ["pserve"]
