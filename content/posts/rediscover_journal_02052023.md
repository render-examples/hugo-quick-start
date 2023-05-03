---
title: "Rediscover journal(4May23) : Deploying MLFlow into containers"
date: 2023-05-04T10:15:27+08:00
tags: ['blog']
---

Hi! If you’re reading this, welcome! This is my personal journalling during my job search journey by writing down what I’m currently working on and learning, especially relating to computer science-y stuff like mlops, VR and more. To learn more about my backstory, please navigate to “Me today”. If you like this article, give it a share on your linkedin! It’ll help me a ton with my job search. Thank you!

One of the first thing prior to moving everything into kubernetes is dockerizing the entire workflow. For Week 1 of May, I have managed to do the following: 

write up a dockerfile setup for the existing mlflow tracking server setup. Note that currently, all experiments and trained models are stored within a ‘.db’ file, which although is a local storage, can be converted into other form of storage provider in the future, such as Amazon, kafka, or apache spark.

 Some of the issue I ran into really quickly were: 

- Networking
    - The major problem. Mainly due to unfamiliarity. In the past, mlflow tracker has always been activated on a looping address (localhost), and have a host-container port mapping. This does not work that well when i need another container to access the tracker container. There are several ways to go about with this, but the main solution is to create a network within a docker compose setup.
    - You can use “network mode” to specify mlflow to only be allowed to be accessed by a specific container.
    - For my case, I setup a network with a random given subnet address:
    
    ```markdown
    networks:
      mlnet:
        name: "mlnet"
        driver: bridge
        ipam:
          config:
              - subnet: "50.1.1.0/24"
                gateway: "50.1.1.4"
    ```
    
- mounting db as volumes
    - because these databases are storages and not computes, we do not want its states to be etheral, but consistent. To do this, we mount these databases as volumes:
    
    ```markdown
    volumes:
          - "./savedmodel:/mlflow/savedmodel"
          - "../db/mlruns.db:/mlflow/db/mlruns.db"
          - "../db/registered_model.db:/mlflow/db/registered_model.db" 
    ```
    
- learning docker-compose
    - In order to solve the network issue, I decided to combine them both together to be launched together as a series of container using docker compose
- protobuf incompatibility with tensorflow version
    - A slight issue that was faced was the error thrown when using tensorflow 2.10.X. To resolve this, i decided to reuse a golang’s alpine and install protobuf from there.

What’s next

- Next is learning to deploy onto kubernetes using Terraform. Some of the things that I will need to learn and cover is both terraform syntax and kubernetes