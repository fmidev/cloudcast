FROM docker.io/rockylinux/rockylinux:8
# registry.access.redhat.com/ubi8
RUN rpm -ivh https://download.fmi.fi/smartmet-open/rhel/8/x86_64/smartmet-open-release-latest-8.noarch.rpm \
             https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm && \ 
    dnf -y install dnf-plugins-core && \
    dnf config-manager --set-enabled powertools && \
    dnf -y module disable postgresql && \
    dnf -y install python39 python39-devel eccodes libglvnd-glx gdal gdal-devel gcc-c++ && \
    dnf -y clean all

COPY . /opt/app-root/src/cloudcast
WORKDIR /opt/app-root/src/cloudcast
RUN python3.9 -m pip --no-cache-dir install -r requirements.txt
