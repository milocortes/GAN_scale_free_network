# Limpiamos ambiente de trabajo
rm(list = ls())

# Cargamos bibliotecas
library(ggplot2)
library(dplyr)
library(reshape)

# Cargamos los resultados
pendientes <- read.csv("/home/milo/PCIC/Maestría/3erSemestre/gpu_arquitecturas/proyectos/proyecto-02/mnist-gan/BA_GAN_pendientes.csv")

pendiente_cast<-rbind.data.frame(data.frame(modelo = "BA",value = pendientes$BA),data.frame(modelo = "GAN",value = pendientes$GAN))

pendiente_cast <- subset(pendiente_cast,value <3.5)

## Dibujamos densidades
pendiente_cast%>%
  ggplot(aes(x=value, fill=modelo)) +
  geom_density(alpha=0.3)+
  ggtitle("Densidades de los valores estimados de los exponentes")+
  labs(x= "Exponente")+
  theme_minimal()
ggsave("densidades_BA_GAN.eps", device=cairo_ps)

## Prueba de medias
ba <- subset(pendiente_cast,modelo =="BA")
gan <- subset(pendiente_cast,modelo =="GAN")

t.test(ba$value,gan$value)
