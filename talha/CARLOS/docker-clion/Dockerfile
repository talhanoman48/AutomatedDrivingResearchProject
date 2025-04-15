FROM rwthika/ros2-tf:humble-desktop-full-tf2.11.0
LABEL authors="Shengyao Jiang"

RUN apt-get update \
  && apt-get install -y ssh \
      build-essential \
      gcc \
      g++ \
      gdb \
      clang \
      cmake \
      rsync \
      gdbserver \
      gdb \
  && apt-get clean

#RUN ( \
#    echo 'LogLevel DEBUG2'; \
#    echo 'PermitRootLogin yes'; \
#    echo 'PasswordAuthentication yes'; \
#    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
#    echo 'UsePAM no';\
#  ) > /etc/ssh/sshd_config_test_clion \
#  && mkdir /run/sshd
  
RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
    echo 'UsePAM no';\
    
  ) > /etc/ssh/sshd_config
  
  
RUN echo "set disable-randomization off" > /etc/gdb/gdbinit
# Change the password 'password' to something more secure
#RUN useradd -m -s /bin/bash user && yes password | passwd user
#RUN echo "user ALL=(ALL)  ALL" > /etc/sudoers
#RUN echo "source /opt/ros/humble/setup.bash" > /home/user/.bashrc
#CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]
RUN yes password | passwd           
WORKDIR /docker-ros/ws
ENTRYPOINT ["/entrypoint.sh"]


