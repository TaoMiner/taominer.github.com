#! /bin/bash
PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
echo "#############################################################"
echo "# Install Shadowsocks-libev server for CentOS               #"
echo "# Intro: http://sngr.org/go/centos-ss-libev                 #"
echo "# Author: SngrDing<me@sngr.org>                             #"
echo "# Thanks: @m0d8ye <https://twitter.com/m0d8ye>              #"
echo "# Thanks: teddysun<https://teddysun.com/357.html>           #"
echo "#############################################################"
echo ""
echo "Please input password for shadowsocks-libev:"
read -p "(input password):" shadowsockspwd
echo -e "Please input port for shadowsocks-libev [1-65535]:"
read -p "(input port):" shadowsocksport
#yum install -y wget unzip openssl-devel gcc swig python python-devel python-setuptools autoconf libtool libevent
yum install -y unzip gcc automake autoconf libtool make build-essential autoconf libtool
yum install -y curl curl-devel zlib-devel openssl-devel perl perl-devel cpio expat-devel gettext-devel
#yum install -y automake make curl curl-devel zlib-devel openssl-devel perl perl-devel cpio expat-devel gettext-devel
cd /root/
wget --no-check-certificate https://github.com/shadowsocks/shadowsocks-libev/archive/master.zip -O /root/shadowsocks-libev.zip
cd /root/
unzip shadowsocks-libev.zi*
cd /root/shadowsocks-libev-master
./configure
make && make install
mkdir /etc/shadowsocks-libev
cat > /etc/shadowsocks-libev/${shadowsocksport}.json<<-EOF
{
    "server":"0.0.0.0",
    "server_port":${shadowsocksport},
    "local_address":"127.0.0.1",
    "local_port":1080,
    "password":"${shadowsockspwd}",
    "timeout":600,
    "method":"rc4-md5"
}
EOF

/etc/init.d/iptables stop
service iptables stop
chkconfig iptables off  
nohup ss-server -c /etc/shadowsocks-libev/${shadowsocksport}.json >/dev/null 2>&1 &
echo "nohup /usr/local/bin/ss-server -c /etc/shadowsocks-libev/${shadowsocksport}.json >/dev/null 2>&1 & " >> /etc/rc.local
    cd /root/
    # Delete shadowsocks-libev floder
    rm -rf /root/shadowsocks-libev-master/
    # Delete shadowsocks-libev zip file
    rm -f shadowsocks-libev.zip
    IP=$(curl -s -4 icanhazip.com)
    echo ""
    echo "Congratulations, shadowsocks-libev install completed!"
    echo "It will auto start at system boot."
    echo -e "Your Server IP: \033[41;37m ${IP} \033[0m"
    echo -e "Your Server Port: \033[41;37m ${shadowsocksport} \033[0m"
    echo -e "Your Password: \033[41;37m ${shadowsockspwd} \033[0m"
    echo -e "Your Local IP: \033[41;37m 127.0.0.1 \033[0m"
    echo -e "Your Local Port: \033[41;37m 1080 \033[0m"
    echo -e "Your Encryption Method: \033[41;37m rc4-md5 \033[0m"
    echo ""
    echo "if you want add more account , just run again."
    echo "if you want delete account , rm -rf /etc/shadowsocks-libev/PORT.json"
    echo "Go http://sngr.org/go/centos-ss-libev/ for more."
    echo ""
