# Scenarij za zagovor: Priporočilni sistem za filme

## 1. Naslovnica
**Govoriš:**
Pozdravljeni. Sem Miha Gradišar in danes vam bom predstavil svojo seminarsko nalogo, v kateri sem zgradil priporočilni sistem za filme. Zanimalo me je, kako dejansko delujejo sistemi, ki nam vsak dan predlagajo vsebino na spletu, zato sem se odločil, da tak sistem zgradim sam in pogledam pod pokrov.

**Kako:** Začni mirno, brez hitenja. Naredi kratek pogled po komisiji, preden nadaljuješ na naslednji diapozitiv.

## 2. Motivacija
**Govoriš:**
Priporočilni sistemi so danes skoraj povsod. YouTube nam izbere naslednji video, Netflix postavi v vrsto naslednjo serijo, spletna trgovina ugiba, kaj bi kupili. Ves čas me je zanimalo, po kakšnem ključu se program odloči, kaj naj komu pokaže. Namesto da bi o tem samo bral, sem hotel te gradnike zgraditi in razstaviti na koščke.

**Kako:** Povej z malo radovednosti v glasu. Po vprašanju o ključu naredi kratek premor za dramatičen učinek.

## 3. Cilj in trije pristopi
**Govoriš:**
Cilj naloge ni bil zgraditi le enega modela, temveč razumeti celotno sliko. Zato sem zgradil in primerjal tri pristope. Prvi je sodelovalno filtriranje z matrično faktorizacijo, ki predstavlja jedro naloge. Drugi je vsebinski model, ki priporoča izključno po žanrih. Tretji pa je hibridni model, ki oba združi in se najbolj približa temu, kar v resnici uporabljajo velika podjetja.

**Kako:** Naštej tri pristope jasno in razločno. Pri vsakem lahko s prsti nakažeš številko.

## 4. Podatki
**Govoriš:**
Uporabil sem javno zbirko podatkov MovieLens, ki vsebuje približno sto tisoč ocen. Zbral jih je nekaj čez šeststo uporabnikov na skoraj deset tisoč filmih. Izvirnih podatkov nisem spreminjal. Sem pa v zbirko ročno dodal pet umetnih uporabnikov z namerno ozkim in izrazitim okusom. Eden na primer gleda izključno akcijske filme. To sem naredil zato, da vnaprej vem, kaj bi moral model priporočiti, in da takoj vidim razlike v obnašanju sistemov.

**Kako:** Ohranjaj enakomeren tempo. Poudari, da so dodani uporabniki umetni in zakaj si jih dodal – to kaže na tvoj premišljen inženirski pristop.

## 5. Sodelovalno filtriranje (Tehnično jedro)
**Govoriš:**
Osrednji model je matrična faktorizacija, natančneje algoritem SVD. Ta model vsakemu uporabniku in filmu pripiše kratek vektor latentnih faktorjev. Te faktorje si lahko predstavljamo kot skrite poteze okusa – na primer nagnjenost k akciji ali drami – ki si jih model izmisli in prilagodi povsem sam. Napovedano oceno za film nato izračunamo kot skalarni produkt teh dveh vektorjev. Model se vektorjev nauči tako, da poskuša zmanjšati napako pri ocenah, ki jih že poznamo.

**Kako:** To je najbolj tehničen del. Upočasni. Ob omembi skalarnega produkta pokaži, da razumeš – to ni le ugibanje, ampak matematika (množenje vektorja uporabnika z vektorjem filma).

## 6. Vrednotenje
**Govoriš:**
Za objektivno primerjavo sem podatke razdelil na učno in testno množico. Uporabil sem dve meri. Glavna je RMSE, torej koren povprečne kvadratne napake, ki meri, za koliko se napovedana ocena v povprečju zgreši od dejanske. Tukaj je nižja vrednost boljša. Druga mera pa je preciznost pri 10. Ta nam pove, koliko od prvih desetih priporočenih filmov je uporabnik dejansko ocenil z vsaj štirimi zvezdicami.

**Kako:** Kratko in stvarno. Pomembno je, da točno veš, kaj pomeni kratica RMSE.

## 7. Napoved ocen (Prvi rezultati)
**Govoriš:**
Poglejmo rezultate. Pri napovedi ocen je matrična faktorizacija brez težav premagala preprosta izhodišča. Dosegla je napako 0,877, medtem ko je najmočnejše izhodišče – to je golo povprečje uporabnika – doseglo 0,943. Razlika je približno sedemodstotna. S tem sem dokazal, da se model iz podatkov nauči dejanskih vzorcev in strukture okusa, ne le ugibanja povprečij.

**Kako:** Pokaži na graf. Govori samozavestno, saj gre za uspešen rezultat tvojega modela.

## 8. Presenetljiv obrat pri razvrščanju
**Govoriš:**
Nato pa me je pričakalo presenečenje. Pri razvrščanju najboljših desetih priporočil je moj napredni model izgubil. Premagalo ga je najpreprostejše izhodišče, ki vsem ponudi samo najbolj priljubljene filme. Sprva je to delovalo kot napaka, a razlog tiči v sami zasnovi mere. Ko ocene za test izločamo naključno, najbolj ocenjevani filmi pogosteje pristanejo v testni množici. Sistem, ki priporoča priljubljenost, zato zmaga skoraj sam od sebe. Model SVD pa meri osebno kakovost, kar sta dve različni stvari.

**Kako:** To je zanimiv in zrel obrat. Pokaži, da razumeš problematiko pristranskosti podatkov. Ne zveni razočarano, ampak analitično.

## 9. Žanrska slepota
**Govoriš:**
Drugo veliko odkritje se je pokazalo pri testiranju z umetnimi uporabniki. Uporabniku, ki je ocenil izključno akcijske filme, je sodelovalno filtriranje vrnilo filme, kot sta Fight Club in The Shawshank Redemption. To so sicer odlični filmi, a niso akcija. Razlog je v tem, da ta model dejansko nikoli ne prebere žanra. Uči se le iz matrike, kdo je kaj ocenil. Ker pa na majhni zbirki prevladujejo splošno priljubljeni filmi, te povozijo žanr. Model je torej žanrsko slep.

**Kako:** Upočasni. Po "niso akcija" naredi kratek premor. Ta diapozitiv kaže tvoje globoko razumevanje omejitev posameznih algoritmov.

## 10. Hibrid kot rešitev
**Govoriš:**
To pomanjkljivost rešuje vsebinski model, ki bere samo žanre. Vendar ta nima občutka za splošno kakovost, zato vrača precej neznane naslove. Zato sem zgradil hibridni model, ki ocene obeh združi z utežjo. Šele ta je akcijskemu uporabniku predlagal filme, ki so akcijski in hkrati visoko cenjeni, na primer The Road Warrior in Goldfinger. To dokazuje, da se resnični sistemi ne zanašajo le na en sam algoritem, temveč kombinirajo njihove prednosti.

**Kako:** Razreši napetost. Sklepni, samozavesten in inženirski ton.

## 11. Omejitve in realnost
**Govoriš:**
Pomembno se je zavedati tudi omejitev. Prva je problem hladnega zagona – nov uporabnik ali film brez ocen nima ustvarjenega vektorja, zato mu sistem ne more napovedati ničesar. Druga je majhnost zbirke podatkov, zaradi katere priljubljenost prehitro preglasi osebni okus. Tretja pa je, da se ta model uči na zamrznjenih podatkih, medtem ko se Netflix ali YouTube učita sproti in dinamično.

**Kako:** Govori stvarno in pošteno. Priznavanje omejitev je znak zrelega inženirja in preprečuje neprijetna vprašanja komisije.

## 12. Zaključek
**Govoriš:**
Glavni vtis te naloge je, da za dobrimi priporočili ne stoji en sam super pameten algoritem, ampak več preprostih, ki se vsak po svoje zmotijo, skupaj pa se odlično dopolnijo. Sodelovalno filtriranje dobro ujame kakovost, a je slepo za žanr; vsebinski model vidi žanr, a ne pozna kakovosti. Hibrid pa uskladi oboje. Najbolj me sedaj zanima, kako daleč bi prišel hibrid na masovni zbirki podatkov.
Zahvaljujem se vam za pozornost in z veseljem odgovorim na vaša vprašanja.

**Kako:** Umirjen zaključek. Pri zadnjem stavku ohrani očesni stik s komisijo, se rahlo nasmehni in počakaj na vprašanja.

---

# Tehnična priloga: Koncepti in razlaga grafov

### 1. Matrična faktorizacija (Matrix Factorization)
* **Koncept:** Ogromna matrika (uporabniki x filmi) z ocenami, kjer je večina celic praznih.
* **Algoritem (SVD):** Razbije redko (sparse) matriko na dve manjši matriki.
* **Latentni faktorji:** Skrite lastnosti (npr. "stopnja akcije"), ki jih model sam odkrije.
[Image of matrix factorization for recommender systems]

### 2. Skalarni produkt (Dot Product)
* **Razumevanje:** Napoved ocene = skalarni produkt vektorja uporabnika in vektorja filma. Če se ujemata (oba imata visoko vrednost za akcijo), je napoved visoka.

### 3. Evalvacijske metrike
* **RMSE:** Meri natančnost ocene (za koliko se model zmoti). Nižje je bolje.
* **Preciznost pri 10:** Ali je med prvimi desetimi filmi vsaj kakšen, ki je uporabniku res všeč (> 4 zvezdice).

### 4. Hibridni sistemi
* **Sodelovalno filtriranje:** "Ljudje so gledali to..." (Kakovost, a žanrsko slepo).
* **Vsebinski model:** "Gledal si akcijo, zato ti priporočam akcijo..." (Natančnost žanra, a slaba kakovost).
* **Hibrid:** Združitev obeh za izkoristek prednosti.
[Image of hybrid recommender system architecture]

### 5. Problem hladnega zagona (Cold Start)
* **Razumevanje:** Kaj ko pride nov uporabnik? Ni podatkov za vektor.
* **Rešitev:** Uporaba popularnih filmov ali vprašalnik ob registraciji.

---

### Interpretacija grafov

* **Slika 1 (RMSE SVD vs Izhodišča):** Modri stolpec (SVD) je najnižji = SVD se dejansko uči vzorcev, ne le povprečij.
* **Slika 2 (RMSE vs k):** Napaka je najnižja pri $k=10$. Višji $k$ vodi v *overfitting*.
* **Slika 3 (RMSE vs uporabniki):** Več podatkov = manjša napaka (sistem je stabilen).
* **Slika 4 (Preciznost pri 10):** SVD "izgubi" proti priljubljenosti. **Razlaga:** Ni napaka modela, ampak lastnost metrike, saj je testna množica pristranska do priljubljenosti.

---

*Nasvet za zagovor: Če te vprašajo o matematiki, reci: "Model s pomočjo optimizacijskega algoritma (npr. SGD - Stochastic Gradient Descent) iterativno popravlja vrednosti vektorjev tako dolgo, dokler se napovedane ocene ne približajo dejanskim ocenam iz učne množice."*
