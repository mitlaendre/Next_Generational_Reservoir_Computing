import numpy as np


# Van OsszesHely darab slot, beleteszünk DarabSzam darab megegyező babot, az összes lehetséges módon végigmegy. Mindegyik egyszer lesz.
# Eredménye vektor: az i. helyen lévő k érték === az i-dik bab (balról jobbra haladva) k-dik slotban van (balról jobbra számolva)
# a "hanyadik" paraméter 1-től kezdődik?
def kombinaciok(DarabSzam = 0, OsszesHely = 0, hanyadik = 0):
    if hanyadik > kombinaciokSzama(DarabSzam, OsszesHely):
        return np.array([])
    if DarabSzam == 1:
        return np.array([hanyadik])

    kombinacio = np.full(DarabSzam,0,dtype=int)
    for i in range(OsszesHely - DarabSzam+1):
        kombikszama = kombinaciokSzama(DarabSzam - 1, OsszesHely - i-1)
        if hanyadik <= kombikszama:
            kombinacio[0] = i+1
            seged = kombinaciok(DarabSzam-1, OsszesHely - i-1, hanyadik)
            for j in range(seged.size):
                kombinacio[j + 1] = seged[j] + i+1
            return kombinacio
        else:
            hanyadik -= kombikszama
def kombinaciokSzama(Darabszam = 0, Osszeshely = 0):
    if Darabszam > Osszeshely:
        return 0
    if Darabszam == Osszeshely:
        return 1

    ertek = 1
    for i in range(Darabszam + 1, Osszeshely + 1):
        ertek *= i

    seged = 1
    for i in range(1, Osszeshely - Darabszam + 1):
        seged *= i

    ertek = int(ertek / seged)
    return ertek



# Van HalmazokSzama darab halmaz, bele elosztunk Elosztogatando darab megegyező "babot", az összes lehetséges módon végigmegy. Mindegyik egyszer lesz.
# Eredménye vektor: az i. helyen lévő k érték === az i-dik halmazban k darab bab van.
# a "hanyadik" paraméter 1-től kezdődik?
def halmazokbaOsztas(HalmazokSzama = 0, Elosztogatando = 0, hanyadik = 0):
    eredmeny = kombinaciok(HalmazokSzama-1,Elosztogatando+HalmazokSzama-1,hanyadik)
    eredmeny1 = np.full(HalmazokSzama,0,dtype=int)
    for i in range(eredmeny.shape[0]-1):
        eredmeny1[i+1] = eredmeny[i+1]-eredmeny[i]-1
    eredmeny1[0] = eredmeny[0]-1
    eredmeny1[eredmeny1.shape[0]-1] = (Elosztogatando+HalmazokSzama-1) - eredmeny[eredmeny.shape[0]-1]
    return eredmeny1
def halmazokbaOsztasokSzama(HalmazokSzama=0, Elosztogatando=0):
    return kombinaciokSzama(HalmazokSzama-1,Elosztogatando+HalmazokSzama-1)



# Van egy szamjegyekMaximumai.size számjegyből álló szám. Minden számjegye bejárhatja az adott helyi értékre megadott maximumot. Az összes így eltárolható számon végigmegy, mindegyiken egyszer.
# Eredménye vektor: az i. helyen lévő k érték ===  az i-dik helyi értéken k érték áll.
# PL: a [10,10,10,10] végigmegy szokásosan a 0-9999 számokon, 10-es számrendszerben (minden helyi érték 10 különböző értéket tárolhat)
# a "hanyadikSzam" paraméter 0-tól kezdődik

def kulonbozoSzamjegyuSzam( szamjegyekMaximumai = np.array([],dtype=int) , hanyadikSzam = 0):
    helyiErtekNagysaga = int(1)
    for i in range(szamjegyekMaximumai.shape[0]):
        helyiErtekNagysaga *= szamjegyekMaximumai[i]

    eredmeny = np.full(szamjegyekMaximumai.size,0)
    for hanyadikSzamjegy in range(szamjegyekMaximumai.size):
        helyiErtekNagysaga = int(helyiErtekNagysaga/szamjegyekMaximumai[hanyadikSzamjegy])
        while(hanyadikSzam >= helyiErtekNagysaga):
            hanyadikSzam -= helyiErtekNagysaga
            eredmeny[hanyadikSzamjegy]+=1


    return eredmeny
def kulonbozoSzamjegyuSzamokSzama( szamjegyekMaximumai = np.array([],dtype=int) ):
    szamolo = 1
    for i in range(szamjegyekMaximumai.size):
        szamolo *= szamjegyekMaximumai[i]
    return szamolo