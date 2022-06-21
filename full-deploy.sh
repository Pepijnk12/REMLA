#!/usr/bin/env bash
echo "Enable minkube ingress"
minikube addons enable ingress
sleep 5s
echo "Applying KubeCTL deployments"
kubectl apply -f k8s-local-deployment.yaml
echo "Portforwarding system available at: http://$1:8080 (When pods are ready use (kubectl get pods) to check"
kubectl port-forward --namespace=ingress-nginx service/ingress-nginx-controller 8080:80