liczba_przypadkow_w_powiecie=function(powiat) {
  ggplot(
    dataset[powiat_miasto ==powiat],aes(data_pliku,liczba_przypadkow,group=1))+
    geom_line()
}


liczba_przypadkow_w_powiecie_liczby=function(powiat,x) {
    dataset[powiat_miasto==powiat][,c('data_pliku','liczba_przypadkow')][order(data_pliku,decreasing = TRUE)][1:x,]
}




najszybszy_wzrost_w_ciagu_x_dni_y_powiatow=function(x,y) {
  powiaty=unique(dataset[,c('powiat_miasto','wojewodztwo')][order(powiat_miasto,wojewodztwo)])
  dataset[is.na(liczba_przypadkow)]$liczba_przypadkow=0
  ramka_powiaty=data.table()
  for (p in 1:nrow(powiaty)) {
    podzbior=dataset[powiat_miasto==powiaty[p,]$powiat_miasto & wojewodztwo==powiaty[p,]$wojewodztwo][order(data_pliku,decreasing = TRUE)]
    minimalna_r_dni=min(abs(podzbior$data_pliku-(podzbior$data_pliku[1]-x)))
    data_odniesienia=podzbior[abs(data_pliku-(podzbior$data_pliku[1]-x))==minimalna_r_dni]$data_pliku
    pr_wzrostu=(podzbior$liczba_przypadkow[1]-podzbior[data_pliku==data_odniesienia]$liczba_przypadkow)/podzbior[data_pliku==data_odniesienia]$liczba_przypadkow
    roznica=podzbior$liczba_przypadkow[1]-podzbior[data_pliku==data_odniesienia]$liczba_przypadkow
    ramka_powiaty=rbind(ramka_powiaty,data.table(powiat=powiaty[p,]$powiat_miasto,
                                                 wojewodztwo=powiaty[p,]$wojewodztwo,
                                                 pr_wzrostu,
                                                 roznica))
  }
  ramka_powiaty=ramka_powiaty[order(pr_wzrostu,roznica,decreasing=TRUE)]
  return(ramka_powiaty[1:y,])
}

najszybszy_wzrost_w_ciagu_x_dni_y_powiatow(30,10)


liczba_przypadkow_w_powiecie('lipski')
liczba_przypadkow_w_powiecie_liczby('lipski',10)


unique(sort(dataset$powiat_miasto))


