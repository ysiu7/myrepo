library(readxl)
library(tidyverse)
library(data.table)
library(stringr)

files <- list.files(path="C:/Users/lenovo/Downloads/danehistorycznepowiaty", pattern="*.txt", full.names=TRUE, recursive=FALSE)


setwd("C:/Users/lenovo/Downloads/danehistorycznepowiaty")

file_list <- list.files(path="C:/Users/lenovo/Downloads/danehistorycznepowiaty",full.names = TRUE,pattern="*.csv")

dataset <- data.frame()


for (i in 1:length(file_list)){
  temp_data <- fread(file_list[i]) 
  napis=substr(file_list[i],50,57)
  temp_data$data_pliku=paste0(substr(napis,1,4),'-',substr(napis,5,6),'-',substr(napis,7,8))
  dataset <- rbind(dataset, temp_data,fill=TRUE)  
}
sort(unique(dataset$powiat_miasto))



dataset$wojewodztwo=str_replace(dataset$wojewodztwo, "Ĺ‚", "ł")
dataset$powiat_miasto=str_replace(dataset$powiat_miasto, "Ĺ‚", "ł")
dataset$wojewodztwo=str_replace(dataset$wojewodztwo, "Ĺ›", "ś")
dataset$powiat_miasto=str_replace(dataset$powiat_miasto, "Ĺ›", "ś")
dataset$wojewodztwo=str_replace(dataset$wojewodztwo, "Ä…", "ą")
dataset$powiat_miasto=str_replace(dataset$powiat_miasto, "Ä…", "ą")
dataset$wojewodztwo=str_replace(dataset$wojewodztwo, "Ăł", "ó")
dataset$powiat_miasto=str_replace(dataset$powiat_miasto, "Ăł", "ó")
dataset$wojewodztwo=str_replace(dataset$wojewodztwo, "Ä™", "ę")
dataset$powiat_miasto=str_replace(dataset$powiat_miasto, "Ä™", "ę")
dataset$wojewodztwo=str_replace(dataset$wojewodztwo, "Ĺ„", "ń")
dataset$powiat_miasto=str_replace(dataset$powiat_miasto, "Ĺ„", "ń")
dataset$wojewodztwo=str_replace(dataset$wojewodztwo, "Ĺ„", "ń")
dataset$powiat_miasto=str_replace(dataset$powiat_miasto, "Ĺ„", "ń")
dataset$wojewodztwo=str_replace(dataset$wojewodztwo, "ĹĽ", "ż")
dataset$powiat_miasto=str_replace(dataset$powiat_miasto, "ĹĽ", "ż")
dataset$wojewodztwo=str_replace(dataset$wojewodztwo, "Ĺş", "ź")
dataset$powiat_miasto=str_replace(dataset$powiat_miasto, "Ĺş", "ź")
dataset$wojewodztwo=str_replace(dataset$wojewodztwo, "Ĺ»", "Ż")
dataset$powiat_miasto=str_replace(dataset$powiat_miasto, "Ĺ»", "Ż")
dataset$wojewodztwo=str_replace(dataset$wojewodztwo, "Ĺ\u0081", "Ł")
dataset$powiat_miasto=str_replace(dataset$powiat_miasto, "Ĺ\u0081", "Ł")
dataset$wojewodztwo=str_replace(dataset$wojewodztwo, "Ĺš", "Ś")
dataset$powiat_miasto=str_replace(dataset$powiat_miasto, "Ĺš", "Ś")
dataset$wojewodztwo=str_replace(dataset$wojewodztwo, "Ĺ‚", "ł")
dataset$powiat_miasto=str_replace(dataset$powiat_miasto, "Ĺ‚", "ł")
dataset$wojewodztwo=str_replace(dataset$wojewodztwo, "Ä‡", "ć")
dataset$powiat_miasto=str_replace(dataset$powiat_miasto, "Ä‡", "ć")



dataset[powiat_miasto=='strzelecki']

names(dataset)
table(dataset[is.na(liczba_przypadkow)]$data_pliku)
sum(is.na(dataset$liczba_przypadkow)) ##102

table(dataset[is.na(liczba_na_10_tys_mieszkancow)]$data_pliku)
sum(is.na(dataset$liczba_na_10_tys_mieszkancow)) ##102

table(dataset[is.na(zgony)]$data_pliku)
sum(is.na(dataset$zgony)) ##2054

table(dataset[is.na(zgony_w_wyniku_covid_bez_chorob_wspolistniejacych)]$data_pliku)
sum(is.na(dataset$zgony_w_wyniku_covid_bez_chorob_wspolistniejacych)) ##2617

table(dataset[is.na(zgony_w_wyniku_covid_i_chorob_wspolistniejacych)]$data_pliku)
sum(is.na(dataset$zgony_w_wyniku_covid_i_chorob_wspolistniejacych)) ##2194

table(dataset[is.na(liczba_zlecen_poz)]$data_pliku)
sum(is.na(dataset$liczba_zlecen_poz)) ##1229

table(dataset[is.na(liczba_osob_objetych_kwarantanna)]$data_pliku)
sum(is.na(dataset$liczba_osob_objetych_kwarantanna)) ##381

table(dataset[is.na(liczba_wykonanych_testow)]$data_pliku)
sum(is.na(dataset$liczba_wykonanych_testow)) ##382

table(dataset[is.na(liczba_testow_z_wynikiem_pozytywnym)]$data_pliku)
sum(is.na(dataset$liczba_testow_z_wynikiem_pozytywnym)) ##382

table(dataset[is.na(liczba_testow_z_wynikiem_negatywnym)]$data_pliku)
sum(is.na(dataset$liczba_testow_z_wynikiem_negatywnym)) ##382

table(dataset[is.na(liczba_pozostalych_testow)]$data_pliku)
sum(is.na(dataset$liczba_pozostalych_testow)) ##382

table(dataset[is.na(stan_rekordu_na)]$data_pliku)
sum(is.na(dataset$stan_rekordu_na)) ## 13335

table(dataset[is.na(liczba_ozdrowiencow)]$data_pliku)
sum(is.na(dataset$liczba_ozdrowiencow)) ## 11839
