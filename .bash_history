C:\Window\System32\wsl.exe -l -v
wsl -l -v
sudo apt update && sudo apt -y upgrade
sudo apt -y install build-essential git curl wget nano
echo '[interop]
appendWindowsPath=false' | sudo tee /etc/wsl.conf
wsl --shutdown
wsl -d Ubuntu-20.04
sudo apt -y install locales
sudo locale-gen en_US.UTF-8
export LANG=en_US.UTF-8
sudo sh -c 'echo "deb https://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt -y install ros-noetic-desktop-full
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
sourfce ~/.bashrc
source ~/.bashr
source ~/.bashrc
cd ~
sudo apt update
hyeongseon@Na:/mnt/c/WINDOWS/system32$ cd ~                                               hyeongseon@Na:~$ sudo apt update                                                          Ign:1 https://packages.ros.org/ros/ubuntu focal InRelease                                 Hit:2 http://archive.ubuntu.com/ubuntu focal InRelease                                    Hit:3 http://security.ubuntu.com/ubuntu focal-security InRelease                          Hit:4 http://archive.ubuntu.com/ubuntu focal-updates InRelease                            Err:5 https://packages.ros.org/ros/ubuntu focal Release                                     Certificate verification failed: The certificate is NOT trusted. The name in the certificate does not match the expected.  Could not handshake: Error in the certificate verification. [IP: 140.211.166.134 443]                                                            Hit:6 http://archive.ubuntu.com/ubuntu focal-backports InRelease                          Reading package lists... Done                                                             E: The repository 'https://packages.ros.org/ros/ubuntu focal Release' does not have a Release file.                                                                                 N: Updating from such a repository can't be done securely, and is therefore disabled by default.                                                                                    N: See apt-secure(8) manpage for repository creation and user configuration detailshyeongseon@Na:/mnt/c/WINDOWS/system32$ cd ~                                               hyeongseon@Na:~$ sudo apt update                                                          Ign:1 https://packages.ros.org/ros/ubuntu focal InRelease                                 Hit:2 http://archive.ubuntu.com/ubuntu focal InRelease                                    Hit:3 http://security.ubuntu.com/ubuntu focal-security InRelease                          Hit:4 http://archive.ubuntu.com/ubuntu focal-updates InRelease                            Err:5 https://packages.ros.org/ros/ubuntu focal Release                                     Certificate verification failed: The certificate is NOT trusted. The name in the certificate does not match the expected.  Could not handshake: Error in the certificate verification. [IP: 140.211.166.134 443]                                                            Hit:6 http://archive.ubuntu.com/ubuntu focal-backports InRelease                          Reading package lists... Done                                                             E: The repository 'https://packages.ros.org/ros/ubuntu focal Release' does not have a Release file.                                                                                 N: Updating from such a repository can't be done securely, and is therefore disabled by default.                                                                                    N: See apt-secure(8) manpage for repository creation and user configuration details
sudo apt -y install ca-certificates gnupg2 lsb-release software-properties-common
sudo rm -f /etc/spt/sources.list.d/ros*.list
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \ | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
sudo rm -f /etc/apt/sources.list.d/ros*.list
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/ros1-latest.list
sudo apt update
sudo apt -y install ros-noetic-desktop-full
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt -y install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool
roscore
rosversion -d
uname -a
lsb_release -a
sudo apt update && sudo apt upgrade -y
sudo apt install -y wireguard
sudo apt update && sudo apt upgrade -y
ls
micromamba activate ros1
wsl -l -v
rosnode list
roscore
sudo apt update && sudo apt -y upgrade
sudo apt -y install wireguard
sudo apt install ros-noetic-desktop-full
source /opt/ros/noetic/setup.bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
sudo systemctl enable --now tailscaled
sudo tailscaled &
sudo tailscale up
tailscale status
ls
dpkg -l | awk '/^ii\s+ros-/{print $2}' | head
dpkg -l | awk '/^ii\s+ros-/{split($2,a,"-"); print a[2]}' | sort -u
echo $ROS_DISTRO
printenv | grep -i ^ROS_
command -v roscore
rosversion -d
roscore
uname -r
sudo apt update
sudo apt install -y git openssh-client
git config --global user.name "hyeongseon"
git config --global user.email "skgudtjs1@gmail.com"
git config --global core.autocrlf input
git config --global core.filemode false
ls -l ~/.ssh
ssh-keygen -t ed25519 -C "skgudtjs1@gmail.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub | clip.exe
cat ~/.ssh/id_ed25519.pub
ssh -T git@github.com
mkdir -p ~/catkin_ws/src
cd ~/catkin_Ws
cd ~/catkin_ws
echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc
echo 'source /catkin_ws/devel/setup.bash' >> ~/.bashrc
cd ~/catkin_ws/src
git clone git@github.com:<USER>/<REPO>.git
git clone git@github.com:baqu4-2025-driverless/baqu4_morai_ws.git
rosdep update
sudo rosdep init
rosdep update
sudo rosdep init
echo $ROS_DISTRO
[ -f ~/catkin_ws/~ ] && rm -i ~/catkin_ws/~ || true
ls
sed -i '/source \/catkin_ws\/devel\/setup\.bash/d' ~/.bashrc
grep -q 'source ~/catkin_ws/devel/setup.bash' ~/.bashrc || \
source ~/.bashrc
cd ..
source ~/.bashrc
cd ..
source ~/.bashrc
ls
cd catkin_ws/
ls
cd src
ls
cd baqu4_morai_ws/
ls
cd ~
mv ~/catkin_ws/src/baqu4_morai_ws ~/baqu4_morai_ws
cd ~/baqu4_morai_ws
rosdep install --from-paths src -r -y
catkin_make
grep -q 'source /opt/ros/noetic/setup.bash' ~/.bashrc || echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc
sed -i '/catkin_ws\/devel\/setup\.bash/d' ~/.bashrc
grep -q 'baqu4_morai_ws/devel/setup.bash' ~/.bashrc || echo '[ -f ~/baqu4_morai_ws/devel/setup.bash ] && source ~/baqu4_morai_ws/devel/setup.bash' >> ~/.bashrc
source ~/.bashrc
rosdep install --from-paths src -r -y
catkin_make
echo $ROS_DISTRO
echo $ROS_PACKAGE_PATH | tr ':' '\n'
rospack list | grep -i baqu4
find src -maxdepth 2 -name package.xml -printf '%h\n'
for d in $(find src -maxdepth 2 -name package.xml -printf '%h\n'); do basename "$d"; done
git submodule status
git submodule update --init --recursive
for d in $(find src -maxdepth 2 -name package.xml -printf '%h\n'); do basename "$d"; done
rosdep install --from-paths src -r -y
catkin_make
ls
cd catkin_Ws
cd catkin_ws
ls
cd src
cd
ls
cd baqu4_morai_ws/
git checkout -b fix/ros1-morai-msgs
git submodule deinit -f src/MORAI-ROS2_morai_msgs
git rm -f src/MORAI-ROS2_morai_msgs
rm -rf .git/modules/src/MORAI-ROS2_morai_msgs
git submodule add git@github.com:MORAI-Autonomous/MORAI-ROS_morai_msgs.git src/morai_msgs
git submodule update --init --recursive
find src/morai_msgs -maxdepth 1 -name package.xml -print
rosdep install --from-paths src -r -y
catkin_make
source ~/.bashrc
echo $ROS_PACKAGE_PATH | tr ':' '\n'
rospack profile
rospack find morai_msgs
rosmsg list | grep -i morai
git status
git add .gitmodules src
git commit -m "replace ROS2 msgs with ROS1 MORAI-ROS_morai_msgs (Noetic)"
git push -u origin HEAD
ls
cd baqu4_morai_ws/
ls
cd src
ls
cd morai_msgs/
ls
cd msg
ls
roslaunch morai_vehicle_control vehicle_ctrl_params.launch
cd baqu4_morai_ws/
ls
code .
cd ..
sudo snap install code --classic
sudo systemctl start snapd.service
sudo service snapd start
source /opt/ros/noetic/setup.bash
ls
cd baqu4_morai_ws/
ls
source /opt/ros/noetic/setup.bash
cd baqu4_morai_ws/
cd
source /opt/ros/noetic/setup.bash
cd baqu4_morai_ws/
source devel/setup.bash
roslaunch morai_vehicle_control vehicle_ctrl_params.launch
ls
cd src
ls
cd morai_msgs/
ls
cd ..
git clone https://github.com/MORAI-EDU/beginner_tutorials_answer.git
cd ..  # src 폴더에서 워크스페이스 루트로 이동
catkin_make
source devel/setup.bash
roslaunch morai_vehicle_control vehicle_ctrl_params.launch
cd ~/baqu4_morai_ws/src
rm -r beginner_tutorials_answer
rm -rf beginner_tutorials_answer
ls
git clone https://github.com/MORAI-EDU/morai_vehicle_control.git
ls
cd ..
ls
rosversion -d
source /opt/ros/noetic/setup.bash
sudo apt install python-rosdep
sudo rosdep init
sudo apt install python3-rosdep python3-rosinstall python33-rosinstall-generator python3-wstool build-essential
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo apt install python3-rosdep
sudo rosdep init
sudo apt update
sudo apt install -y python3-rosdep
source ~/.bashrc
ls
cd ~/catkin_ws
ls
cd src/
ls
catkin_init_workspace
cd ~/catkin_ws/
catkin_make
cd 
ls
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install curl
sudo apt-get install git
sudo apt-get install new-tools
sudo apt update
sudo apt install -y net-tools
sudo apt-get install python3-pip
pip install scikit-learn==0/24/1
pip install scikit-learn==0.24.1
sudo apt-get install ros-noetic-rosbridge-server
sudo apt-get install ros-noetic-velodyne
ls
cd catkin_ws/src
ls
git clone https://github.com/MORAI-Autonomous/MORAI-ROS_morai_msgs.git
git clone https://github.com/MORAI-EDU/beginner_tutorials_blanks.git
cd ..
catkin_make
cd ~/
mkdir -p catkin_ws/src
cd catkin_ws
catkin_make
cd ~/
cd catkin_ws/src
catkin_create_pkg beginner_tutorials rospy std_msgs
cd beginner_tutorials && mkdir scripts
cd ~/catkin_ws
catkin_make
source ~/catkin_ws/devel/setup.bash
rospack profile
cd ~/catkin_ws/src/beginner_tutorials/scripts
wget https://raw.github.com/ros/ros_tutorials/noetic-devel/rospy_tutorials/001_talker_listener/talker.py
ls
vim talker.py
wget https://raw.github.com/ros/ros_tutorials/noetic-devel/rospy_tutorials/001_talker_listener/listener.py
ls
vim listener.py 
cd
clear
roscore
hyeongseon@Na:~$ roscore
... logging to /home/hyeongseon/.ros/log/80064992-8ef4-11f0-8b92-00155d0ede35/roslaunch-Na-11164.log
Checking log directory for disk usage. This may take a while.
Press Ctrl-C to interrupt
Done checking log file disk usage. Usage is <1GB.
started roslaunch server http://Na:38017/
ros_comm version 1.17.4
SUMMARY
========
PARAMETERS
NODES
auto-starting new master
process[master]: started with pid [11172]
ROS_MASTER_URI=http://Na:11311/
setting /run_id to 80064992-8ef4-11f0-8b92-00155d0ede35
Traceback (most recent call last):
RuntimeError: Multiple packages found with the same name "beginner_tutorials":
- beginner_tutorials
- beginner_tutorials_blanks
[master] killing on exit
cd ~/baqu4_morai_ws/src
grep -R --include=package.xml -n '<name>beginner_tutorials</name>' .
exec bash -l
source /opt/ros/noetic/setup.bash
[ -f ~/baqu4_morai_ws/devel/setup.bash ] && source ~/baqu4_morai_ws/devel/setup.bash
hyeongseon@Na:~/baqu4_morai_ws/src$ exec bash -l
hyeongseon@Na:~/baqu4_morai_ws/src$ source /opt/ros/noetic/setup.bash
hyeongseon@Na:~/baqu4_morai_ws/src$ [ -f ~/baqu4_morai_ws/devel/setup.bash ] && source ~/baqu4_morai_ws/devel/setup.bash
hyeongseon@Na:~/baqu4_morai_ws/src$
ls -l ~/baqu4_morai_ws/devel/setup.bash
cd ~/baqu4_morai_ws
catkin_make
# 잘못된/중복된 설정 줄 제거
sed -i '\|/opt/ros/noetic/setup.bash|d' ~/.bashrc
sed -i '\|baqu4_morai_ws/devel/setup.bash|d' ~/.bashrc
sed -i '/\/devel\/setup\.bash/d' ~/.bashrc   # 혹시 다른 워크스페이스 흔적도 제거
# 올바른 두 줄만 추가
cat >> ~/.bashrc <<'EOF'
source /opt/ros/noetic/setup.bash
[ -f ~/baqu4_morai_ws/devel/setup.bash ] && source ~/baqu4_morai_ws/devel/setup.bash
EOF

# 새 로그인 셸로 재적용
exec bash -l
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
rospack find beginner_tutorials
rosrun beginner_tutorials listener.py
ls
cd src/beginner_tutorials/
cd scripts
ls
rosrun beginner_tutorials listener.py
head -1 listener.py
chmod +x listener.py talker.py
rosrun beginner_tutorials talker.py
roscd begginer_tutorials/scripts
ls
cd baqu4_morai_ws/
ls
cd ..
cd catkin_ws/
ls
roscd begginer_tutorials/
roscd beginer_tutorials/
sudo ln -s /usr/bin/python3 /usr/bin/python
roscd beginer_tutorials/
cd src
ls
cd beginner_tutorials
cd scripts
ls
chmod +x talker.py
rosrun beginner_tutorials talker.py
cd ~/catkin_ws/src
touch beginner_tutorials_blanks/CATKIN_IGNORE
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source ~/catkin_ws/devel/setup.bash
rospack find beginner_tutorials
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
rosrun beginner_tutorials talker.py
roslaunch rosbridge_server rosbridge_websocket.launch
echo "$ROS_DISTRO"                      # noetic 이어야 함
echo "$ROS_PACKAGE_PATH" | tr ':' '\n'  # ~/baqu4_morai_ws/src 와 /opt/ros/noetic/share 가 보여야 함
roscore
cd
ifconfig
rostopic list
roslaunch rosbridge_server rosbridge_websocket.launch
rostopic pub /
rostopic pub /ctrl_cmd morai_msgs/CtrlCmd "{longlCmdType: 0, accel: 0.0, brake: 0.0, steering: 0.0, velocity: 0.0, acceleration: 0.0}"
source ~/catkin_ws/devel/setup.bash
rostopic list | grep -E 'image|camera'
rostopic hz /image_jpeg/compressed
ls
cd catkin_ws/src/MORAI-ROS_morai_msgs/msg
cd
rostopic pub /
cd catkin_ws/src/MORAI-ROS_morai_msgs/msg
ls
cd
rostopic pub /ctrl_cmd_0 morai_msgs/CtrlCmd "{longlCmdType: 0, accel: 0.0, brake: 0.0, steering: 0.0, velocity: 0.0, acceleration: 0.0}" 
roslaunch rosbridge_server rosbridge_websocket.launch
source ~/.bashrc
cd ~/catkin_ws/
cd
source ~/catkin_ws/devel/setup.bash
roscore
source ~/catkin_ws/devel/setup.bash
roslaunch rosbridge_server rosbridge_websocket.launch
code .
ls
cd catkin_ws/
ls
cd src
ls
cd ..
cd
rm -rf catkin_ws
nano ~/.bashrc
source ~/.bashrc
rm -rf catkin_ws
git clone https://github.com/minsungk02/catkin_ws.git
ls
cd catkin_ws
WSL_IP=$(hostname -I | awk '{print $1}')
export ROS_MASTER_URI=http://$WSL_IP:11311
export ROS_HOSTNAME=$WSL_IP
echo "ROS_MASTER_URI=$ROS_MASTER_URI  ROS_HOSTNAME=$ROS_HOSTNAME"
grep -q "MORAI_ROS_ENV_BEGIN" ~/.bashrc || cat <<'EOF' >> ~/.bashrc
# ==== MORAI_ROS_ENV_BEGIN ====

WSL_IP=$(hostname -I | awk '{print $1}')
export ROS_MASTER_URI=http://$WSL_IP:11311
export ROS_HOSTNAME=$WSL_IP
echo "[MORAI] ROS_MASTER_URI=$ROS_MASTER_URI  ROS_HOSTNAME=$ROS_HOSTNAME"
# ==== MORAI_ROS_ENV_END ====

EOF

source ~/.bashrc
cd ~/catkin_ws
rm -rf build devel
sudo rosdep init 2>/dev/null || true
source ~/catkin_ws/devel/setup.bash
roscore
source ~/catkin_ws/devel/setup.bashsource ~/catkin_ws/devel/setup.bashclear
clear
source ~/catkin_ws/devel/setup.bash
rostopic list | grep -E 'image|camera'
rostopic hz /image_jpeg/compressed
rosrun image_transport republish compressed
rosrun image_transport republish compressed in:=/image_jpeg raw out:=/camera/image_raw
source ~/catkin_ws/devel/setup.bash
roslaunch rosbridge_server rosbridge_websocket.launch port:=9090
clear
ls
cd catkin_ws/
ls
sudo rosdep init 2>/dev/null || true
rosdep update
ls
rosdep install --from-paths src --ignore-src -r -y
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
ls src/MORAI-ROS_morai_msgs/msg | head
source ~/catkin_ws/devel/setup.bash
roscore
source ~/catkin_ws/devel/setup.bash
roslaunch rosbridge_server rosbridge_websocket.launch port:=9090
rqt_image_view /lane_overlay
source ~/catkin_ws/devel/setup.bash
rostopic list | grep -E 'image|camera'
rostopic hz /image_jpeg/compressed
rostopic list | grep -E 'image|camera'
rostopic hz /image_jpeg/compressed
rosrun image_transport republish compressed in:=/image_jpeg raw out:=/camera/image_raw
rqt_image_view /camera/image_raw
source ~/catkin_ws/devel/setup.bash
rosparam set /lane_tracking/kp 0.35
rosparam set /lane_tracking/roi_y_ratio 0.55
rosparam set /lane_tracking/debug true
rosrun lane_tracking_pkg lane_tracking_node.py
CLEAR
clear
code ~/catkin_ws/src/lane_tracking_pkg/vision/detector.py
nano ~/.bashrc
source ~/.bashrc
code ~/catkin_ws/src/lane_tracking_pkg/vision/detector.py
nano ~/.bashrc
code ~/catkin_ws/src/lane_tracking_pkg/vision/detector.py
source ~/.bashrc
code ~/catkin_ws/src/lane_tracking_pkg/vision/detector.py
nano ~/.bashrc
source ~/.bashrc
code ~/catkin_ws/src/lane_tracking_pkg/vision/detector.py
nano ~/.bashrc
sudo ln -s "/mnt/c/Users/kksnh/AppData/Local/Programs/Microsoft VS Code/Code.exe" /usr/local/bin/code
code ~/catkin_ws/src/lane_tracking_pkg/vision/detector.py
rvice_worker\service_worker_storage.cc:1732] Failed to delete the database: Database IO
cd ~/catkin_ws
catkin_make
source devel/setup.bash
roscore
source ~/catkin_ws/devel/setup.bash
roslaunch rosbridge_server rosbridge_websocket.launch port:=9090
source ~/catkin_ws/devel/setup.bash
rosparam set /lane_tracking/kp 0.35
rosparam set /lane_tracking/roi_y_ratio 0.55
rosparam set /lane_tracking/debug true
rosrun lane_tracking_pkg lane_tracking_node.py
ls
cd catkin_ws
git init
touch .gitignore
nano .gitignore
git add .
git commit -m "Initial commit of my catkin_ws"
git remote add origin https://github.com/Hyeong-Seon/catkin_ws.git
git remote -v
rm -rf .git
git init
git add .
git commit -m "Initial commit on my repository"
git remote add origin https://github.com/Hyeong-Seon/catkin_ws.git
git push -u origin main
git branch -M main
git push -u origin main
cd ~/catkin_ws
git push -u origin master
git branch -M main
git push -u origin main
git pull origin main
git push origin main
git pull origin main
git pull origin main --allow-unrelated-histories
git push origin main
git remote -v
source ~/catkin_ws/devel/setup.bash
rqt_image_view /camera/image_raw
rqt_image_view /lane_overlay
rostopic hz /lane_overlay
source ~/catkin_ws/devel/setup.bash
rostopic list | grep -E 'image|camera'
rostopic hz /image_jpeg/compressed
rqt_image_view /camera/image_raw
rosparam set /lane_tracking/kp 0.35
rosparam set /lane_tracking/roi_y_ratio 0.55
rosparam set /lane_tracking/debug true
rosrun lane_tracking_pkg lane_tracking_node.py
rqt_image_view /camera/image_raw
rostopic echo /image_jpeg/compressed
rostopic hz /image_jpeg/compressed
rqt_image_view /camera/image_raw
rosrun image_transport republish compressed in:=/image_jpeg raw out:=/camera/image_raw
rqt_image_view /camera/image_raw
rosrun image_transport republish compressed in:=/image_jpeg raw out:=/camera/image_raw
source ~/catkin_ws/devel/setup.bash
roslaunch rosbridge_server rosbridge_websocket.launch port:=9090
source ~/catkin_ws/devel/setup.bash
# 파라미터 설정
rosparam set /lane_tracking/kp 0.35
rosparam set /lane_tracking/roi_y_ratio 0.55
rosparam set /lane_tracking/debug true
# 라인트래킹 노드 실행
rosrun lane_tracking_pkg lane_tracking_node.py
source ~/catkin_ws/devel/setup.bash
roscore
source ~/catkin_ws/devel/setup.bash
rostopic list | grep -E 'image|camera'
rostopic hz /image_jpeg/compressed
source ~/catkin_ws/devel/setup.bash
roslaunch rosbridge_server rosbridge_websocket.launch port:=9090
source ~/catkin_ws/devel/setup.bash
roscore
ls
cd catkin_ws/
cd ..
rm catkin_ws
rm -rf catkin_ws
ls
git clone https://github.com/minsungk02/catkin_ws.git catkin_ws
ls
rm -rf catkin_ws
ls
git clone https://github.com/minsungk02/catkin_ws.git catkin_ws
git remote set-url origin https://github.com/Hyeong-Seon/catkin_ws.git
cd catkin_ws/
git remote set-url origin https://github.com/Hyeong-Seon/catkin_ws.git
git push origin main --force
ls
cd catkin_ws/
ls
cd catkin_ws
ls
cd home
code .
cd catkin_ws/
ls
cd src
ls
cd catkin_ws/
ls
rm -rf build devel
sudo rosdep init 2>/dev/null || true
rosdep update
rosdep install --from-paths src --ignore-src -r -y
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
ls src/MORAI-ROS_morai_msgs/msg | head
roscore
source ~/catkin_ws/devel/setup.bash
roslaunch rosbridge_server
cd catkin_ws/
source ~/catkin_ws/devel/setup.bash
roslaunch rosbridge_server
roslaunch rosbridge_server rosbridge_websocket.launch port:=9090
cd catkin_ws/
source ~/catkin_ws/devel/setup.bash
rostopic list | grep -E 'image|camera'
rostopic hz /image_jpeg/compressed
rqt_image_view /camera/image_raw
rosrun image_transport republish compressed in:=/image_jpeg raw out:=/camera/image_raw
cd ~/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
roscore
source ~/catkin_ws/devel/setup.bash
cd catkin_ws/
source ~/catkin_ws/devel/setup.bash
roslaunch rosbridge_server rosbridge_websocket.launch port:=9090
cd catkin_ws/
source ~/catkin_ws/devel/setup.bash
rostopic list | grep -E 'image|camera'
rostopic hz /image_jpeg/compressed
rqt_image_view /camera/image_raw
rosrun perception_pkg traffic_light_color_node.py   _image_topic:=/camera/image_raw   _threshold_ratio:=0.02   _roi_x_start:=0.545 _roi_y_start:=0.0 _roi_width:=0.455 _roi_height:=0.333
ls
cd src
ls
cd perception_pkg/
ls
cd perception_pkg/
ls
cd perception
ls
ls ~/catkin_ws/src/perception_pkg/scripts/traffic_light_color_node.py
cd ~/catkin_ws
git pull origin main
chmod +x ~/catkin_ws/src/perception_pkg/scripts/traffic_light_color_node.py
ls ~/catkin_ws/src/perception_pkg/scripts/traffic_light_color_node.py
rosrun perception_pkg traffic_light_color_node.py   _image_topic:=/camera/image_raw   _threshold_ratio:=0.02   _roi_x_start:=0.545 _roi_y_start:=0.0 _roi_width:=0.455 _roi_height:=0.333
cd catkin_ws/
cd ~/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash   # 새 터미널을 열 때마다 반복
rosrun image_transport republish compressed in:=/image_jpeg raw out:=/camera/image_raw
cd catkin_ws/
source ~/catkin_ws/devel/setup.bash
rostopic echo /traffic_detection        # True면 초록불 인식
rqt_image_view /camera/image_raw
cd catkin_ws/
source ~/catkin_ws/devel/setup.bash
rostopic echo /traffic_detection
ls
cd catkin_ws/
ls
cd ~/catkin_ws
git status
git add .
git commit -m "신호등""
git commit -m "신호등"
git remote add upstream https://github.com/minsungk02/catkin_ws.git
git remote -v
git fetch upstream
git merge upstream/main
git add .
git commit
git push origin main
git add src/perception_pkg/scripts/speed_sign_node.py src/perception_pkg/perception_pkg/perception/object_detection/yolo_speed_sign_pt.py src/perception_pkg/launch/lane_bringup.launch src/perception_pkg/models
git commit -m "Add YOLOv11 speed_sign"
git push origin feature/tl-state
git status -sb
git add src/perception_pkg/scripts/speed_sign_node.py         src/perception_pkg/perception_pkg/perception/object_detection/yolo_speed_sign_pt.py         src/perception_pkg/launch/lane_bringup.launch         src/perception_pkg/models/traffic_sign_detector.pt
git commit -m "Add YOLOv11 speed sign detector and speed range publishing"
git checkout main
git merge feature/tl-state
git push origin main
git fetch origin
git checkout main
git pull --rebase origin main
git push origin main
git status -sb
git add <파일/디렉터리 경로>
git status -sb          # 수정된 파일 목록 확인
git diff                # 마지막 커밋과 비교해 변경 내용 점검
git status
cd ~/catkin_ws/src
git clone --recursive https://github.com/leggedrobotics/darknet_ros.git
cd ~/catkin_ws/src/darknet_ros/darknet_ros/yolo_network_config/weights
wget http://pjreddie.com/media/files/yolov3.weights          # 기본 COCO 용
# 필요하면 VOC 버전도 추가:
wget http://pjreddie.com/media/files/yolov3-voc.weights
cd src/darknet_ros/darknet_ros/yolo_network_config/cfg
ls
cd ../../config
ls
cd ../launch
ls
nano yolo_v3.launch
cd ~/catkin_ws/src/darknet_ros/darknet_ros/config
cp yolov3.yaml yolov3_custom.yaml
nano yolov3_custom.yaml
vim  yolov3_custom.yaml
cd ~/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
roslaunch darknet_ros yolo_v3.launch
ls
cd catkin_ws/
ls
cd external/
ls
cd..
cd ..
git status
git restore --staged .
git checkout -- .
git pull origin main
ls
cd external/
ls
cd obstacle_yolov11/
ls
cd raw
ls
cd images
ls
rm orange*.png
ls
cd ..
ls
cd labels
ls
rm orange*.txt
ls
cd catkin_ws/external/obstacle_yolov11/
ls
cd raw
ls
cd images
ls
cd ~/catkin_ws/external/obstacle_yolov11
git status
git add -A
git rm --cached src/darknet_ros
cd ~/catkin_ws
git status
rm -rf src/darknet_ros/.git
git rm --cached -r src/darknet_ros
git add -A
git commit -m "Remove orange dataset and fix darknet_ros submodule issue"
git push origin main
git add external/obstacle_yolov11/augment_dataset.py         src/control_pkg/scripts/vehicle_control_node.py         src/perception_pkg/scripts/speed_sign_node.py         src/perception_pkg/launch/lane_bringup.launch         external/obstacle_yolov11/dataset         src/darknet_ros         external/obstacle_yolov11/dataset_smoke
