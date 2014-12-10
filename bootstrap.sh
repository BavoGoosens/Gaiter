#!/usr/bin/env bash

sudo apt-get install -y python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose git
if [[ -d "/home/vagrant/scikit-learn/" && ! -L "/home/vagrant/scikit-learn/" ]] ; then
    echo "Already installed"
    cd /home/vagrant/scikit-learn/
    git pull
    #sudo make
else
    git clone git://github.com/scikit-learn/scikit-learn.git
    # add to path
    echo 'export PYTHONPATH=${PYTHONPATH}:/home/vagrant/scikit-learn' >> /home/vagrant/.bashrc
    echo 'export PYTHONPATH=${PYTHONPATH}:/home/vagrant/scikit-learn/sklearn' >> /home/vagrant/.bashrc
    cd /home/vagrant/scikit-learn/
    sudo make
fi


