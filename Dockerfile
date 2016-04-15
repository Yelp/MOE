FROM yelpmoe/min_reqs
MAINTAINER Scott Clark <sclark@yelp.com> and Eric Liu <eliu@yelp.com>

# Install pip systemwide for Python.
ADD https://bootstrap.pypa.io/get-pip.py /tmp/get-pip.py
RUN python /tmp/get-pip.py

# Install python requirements (these should be all in yelpmoe/min_reqs, but it is done again here to be safe)
ADD requirements.txt /home/app/MOE/
RUN cd /home/app/MOE/ && pip install -r requirements.txt

# Copy over the code
ADD . /home/app/MOE/
WORKDIR /home/app/MOE

# Install the python
ENV MOE_NO_BUILD_CPP True
RUN python setup.py install

# Build the C++
WORKDIR /home/app/MOE/moe
RUN rm -rf build
RUN mkdir build
WORKDIR /home/app/MOE/moe/build
RUN cmake /home/app/MOE/moe/optimal_learning/cpp/
RUN make

# Copy the built C++ into the python
RUN cp -r /home/app/MOE/moe/build $(python -c "import site; print(site.getsitepackages()[0])")/moe/.

RUN chown -R app:app /home/app/MOE && chmod -R a+r /home/app/MOE
WORKDIR /home/app/MOE

# Run tests
RUN make test

# Configure docker container.
EXPOSE 6543

# Set up the webserver
USER app
CMD ["development.ini"]
ENTRYPOINT ["pserve"]
