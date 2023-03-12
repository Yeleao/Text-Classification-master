import pkuseg

seg = pkuseg.pkuseg(model_name='default_v2', postag=True)
text = seg.cut('吃了3天的头孢和阿奇，没有好转，医生开了雾化治疗的处方单')
print(text)
