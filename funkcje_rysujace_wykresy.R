liczba_przypadkow_w_powiecie=function(powiat) {
  ggplot(
    dataset[powiat_miasto ==powiat],aes(data_pliku,liczba_przypadkow,group=1))+
    geom_line()
}


liczba_przypadkow_w_powiecie('strzelecki')

unique(sort(dataset$powiat_miasto))


