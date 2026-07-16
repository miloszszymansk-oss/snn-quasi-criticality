# CriticalCortex — Presentation & LinkedIn Article (EN / PL)

Scientifically honest framing throughout: the system self-organizes to **quasi-criticality
(SOqC)** — a stable, sub-critical branching ratio (σ_MR ≈ 0.98) with a **truncated** power law
(τ ≈ 2.2) — *not* mean-field self-organized criticality (τ = 3/2, σ = 1). The rigor of that
distinction is treated as a feature, not a footnote.

---
---

# PART 1 — PRESENTATION OUTLINE

## 🇬🇧 English version (12 slides)

### Slide 1 — Title
**CriticalCortex: Watching the Brain Balance on the Edge of Chaos**
- A real-time engine for Self-Organized Quasi-Criticality (SOqC) in spiking neural networks.
- Rust kernel · custom binary protocol · WebGL. Simulation you can *watch*, science you can *verify*.
- *Visual:* the live 3D neuron sphere (money-shot screenshot), cyan excitatory / red inhibitory.

### Slide 2 — The Question: why "criticality"?
- The **critical-brain hypothesis**: cortex may operate near a phase transition, maximizing dynamic range, information transmission, and susceptibility.
- Signature: **scale-free neuronal avalanches** — bursts of activity with no characteristic size.
- The hard part: distinguishing *true* criticality from something that merely looks power-law-ish.
- *Visual:* sub-critical → critical → super-critical schematic with example raster plots.

### Slide 3 — The Nuance That Drove the Project: SOC vs SOqC
- Mean-field self-organized criticality predicts an exact avalanche exponent **τ = 3/2** and branching ratio **σ = 1**.
- But homeostatic negative feedback (synaptic depression) is known to park a system **just below** the critical point → **self-organized *quasi*-criticality** (Bonachela & Muñoz).
- I set out to show textbook SOC. The data said SOqC. I followed the data.
- *Visual:* two avalanche-size distributions — a straight power law vs a truncated one with a finite-size cutoff.

### Slide 4 — The Model
- Izhikevich spiking neurons, **80/20 excitatory–inhibitory balance**, fixed in-degree connectivity.
- The self-organizing mechanism: **short-term synaptic depression** — a per-neuron resource ⟨x⟩ that depletes on spikes and recovers slowly.
- ⟨x⟩ is the homeostat: it lowers effective gain after bursts, arresting runaway activity.
- *Visual:* network schematic + the ⟨x⟩ deplete-and-recover time course.

### Slide 5 — Architecture: HPC Meets the Browser
- **Rust** zero-allocation simulation kernel (~137k spikes/s) → **Python** orchestration → **binary WebSocket** → **WebGL** frontend.
- Simulation tick and render tick are **decoupled**: many sim steps map to one broadcast frame.
- The kernel is topology-agnostic (reads only CSR arrays) — every experiment reuses it unchanged.
- *Visual:* end-to-end architecture diagram with the data path highlighted.

### Slide 6 — The Custom Binary Protocol
- Hot path is a tightly packed little-endian frame: header + `uint16` spike ids + `uint8` per-neuron ⟨x⟩.
- ~2 KB/s for spikes, ~60 KB/s for the full resource heatmap at N=2000, 30 FPS — the browser never breaks a sweat.
- Bounded memory: the spike buffer is reset every batch, so an indefinite live run never grows.
- *Visual:* the byte-level frame layout + bandwidth/throughput numbers.

### Slide 7 — The Spatial Connectome
- Connectivity probability decays with 3D distance on a Fibonacci sphere; **conduction delays scale with distance**.
- Result: avalanches propagate as **spatially contiguous ripples**, not global flicker.
- Inhibition is interspersed (not clustered), preserving local E/I balance; the Rust kernel needed **zero changes**.
- *Visual:* the sphere with a faint local-edge mesh + a cascade rippling outward.

### Slide 8 — The Money Shot: the ⟨x⟩ Heatmap
- One toggle switches the cloud from spike-glow to a **resource heatmap** (depleted = dark purple → recovered = bright yellow).
- You literally watch an avalanche carve a **"scar" of depletion** that slowly heals as ⟨x⟩ recovers.
- This makes the *invisible* homeostatic feedback — the thing that holds σ ≈ 0.98 — *visible*.
- *Visual:* side-by-side spike mode vs heatmap mode of the same instant.

### Slide 9 — Scientific Validation: No Fitting Artifacts
- Exact discrete **Clauset–Shalizi–Newman** MLE, KS-selected x_min, bootstrap goodness-of-fit + confidence intervals.
- **Multistep-regression (MR) branching estimator** — recovers the true ratio to ±0.005 under 10% subsampling, where the naive estimator reads 0.5.
- The verdict gate is a *classifier*: it passes genuine SOqC and **fails all 7 non-SOqC controls** (critical, dead, runaway, drifting σ…).
- *Visual:* the validation report + the σ_MR(N) plot.

### Slide 10 — Results
- **σ_MR ≈ 0.98**, stable and **N-invariant** (range < 0.003 across N = 500–2000) → self-organization, not coincidence.
- Avalanche law: **truncated power law, τ ≈ 2.2** — reproduced on the spatial connectome.
- Honest verdict: **quasi-critical**, not mean-field critical. The finite-size cutoff and steep exponent are the SOqC signature.
- *Visual:* σ_MR flat vs N; avalanche distribution with its cutoff; mean-field checks failing above the SOqC pass.

### Slide 11 — Why It Matters
- A bridge between **high-performance engineering** and **complex-systems neuroscience** — one reusable, falsifiable pipeline.
- Turns an abstract mechanism (homeostatic self-organization) into something you can *see* and *test* live.
- A template for honest computational science: let the gate fail, then report what's actually true.
- *Visual:* one-line summary + the live demo QR/URL.

### Slide 12 — What's Next / Q&A
- Per-neuron adaptation studies, larger N, finite-size-scaling collapse, alternative homeostatic rules.
- Open question for the room: how far below σ = 1 does biology actually sit — and does it matter?
- *Visual:* invitation to the live demo; contact / repo.

---

## 🇵🇱 Wersja polska (12 slajdów)

### Slajd 1 — Tytuł
**CriticalCortex: Jak mózg balansuje na krawędzi chaosu**
- Silnik czasu rzeczywistego do badania samoorganizującej się kwazi-krytyczności (SOqC) w impulsowych sieciach neuronowych.
- Jądro w Rust · autorski protokół binarny · WebGL. Symulacja, którą można *oglądać*, i nauka, którą można *zweryfikować*.
- *Wizualizacja:* trójwymiarowa sfera neuronów na żywo — cyjan (pobudzające) / czerwień (hamujące).

### Slajd 2 — Pytanie: dlaczego „krytyczność”?
- **Hipoteza krytycznego mózgu**: kora może działać blisko przejścia fazowego, maksymalizując zakres dynamiczny, transmisję informacji i podatność.
- Sygnatura: **bezskalowe lawiny neuronalne** — wyładowania aktywności bez charakterystycznej wielkości.
- Trudność: odróżnić *prawdziwą* krytyczność od czegoś, co jedynie *przypomina* prawo potęgowe.
- *Wizualizacja:* schemat reżimów podkrytyczny → krytyczny → nadkrytyczny z przykładowymi rasterami.

### Slajd 3 — Niuans, który napędził projekt: SOC vs SOqC
- Samoorganizująca się krytyczność (SOC) w polu średnim przewiduje dokładny wykładnik lawin **τ = 3/2** i współczynnik rozgałęzień **σ = 1**.
- Jednak homeostatyczne sprzężenie zwrotne (depresja synaptyczna) ustawia układ **tuż poniżej** punktu krytycznego → **samoorganizująca się *kwazi*-krytyczność** (Bonachela i Muñoz).
- Zamierzałem pokazać podręcznikowe SOC. Dane wskazały SOqC. Poszedłem za danymi.
- *Wizualizacja:* dwa rozkłady wielkości lawin — proste prawo potęgowe vs obcięte, ze skończonym odcięciem.

### Slajd 4 — Model
- Impulsowe neurony Izhikevicha, **równowaga pobudzenie–hamowanie 80/20**, ustalony stopień wejściowy sieci.
- Mechanizm samoorganizacji: **krótkoterminowa depresja synaptyczna** — zasób ⟨x⟩ na neuron, który wyczerpuje się przy impulsach i wolno się odnawia.
- ⟨x⟩ jest homeostatem: po wyładowaniach obniża efektywne wzmocnienie, hamując niekontrolowaną aktywność.
- *Wizualizacja:* schemat sieci + przebieg wyczerpywania i odnowy ⟨x⟩.

### Slajd 5 — Architektura: HPC spotyka przeglądarkę
- Jądro symulacji w **Rust** o zerowej alokacji (~137 tys. impulsów/s) → orkiestracja w **Pythonie** → **binarny WebSocket** → front-end **WebGL**.
- Takt symulacji i takt renderowania są **rozprzężone**: wiele kroków symulacji przypada na jedną ramkę.
- Jądro jest niezależne od topologii (czyta wyłącznie tablice CSR) — każdy eksperyment używa go bez zmian.
- *Wizualizacja:* diagram architektury z podświetloną ścieżką danych.

### Slajd 6 — Autorski protokół binarny
- Gorąca ścieżka to ciasno upakowana ramka little-endian: nagłówek + identyfikatory impulsów `uint16` + ⟨x⟩ na neuron jako `uint8`.
- ~2 KB/s dla impulsów, ~60 KB/s dla pełnej mapy cieplnej zasobów przy N=2000, 30 FPS — przeglądarka nawet nie drgnie.
- Ograniczona pamięć: bufor impulsów jest zerowany co partię, więc nieskończony przebieg nigdy nie rośnie.
- *Wizualizacja:* układ ramki na poziomie bajtów + liczby przepustowości.

### Slajd 7 — Konektom przestrzenny
- Prawdopodobieństwo połączenia maleje z odległością 3D na sferze Fibonacciego; **opóźnienia przewodzenia rosną z odległością**.
- Efekt: lawiny propagują się jako **spójne przestrzennie fale**, a nie globalne migotanie.
- Hamowanie jest rozproszone (nie skupione), co zachowuje lokalną równowagę E/I; jądro Rust nie wymagało **żadnych zmian**.
- *Wizualizacja:* sfera z delikatną siatką lokalnych połączeń + kaskada rozchodząca się na zewnątrz.

### Slajd 8 — Sedno wizualne: mapa cieplna ⟨x⟩
- Jedno przełączenie zmienia chmurę z błysków impulsów na **mapę cieplną zasobów** (wyczerpane = ciemny fiolet → odnowione = jasna żółć).
- Widać dosłownie, jak lawina wypala **„bliznę” wyczerpania**, która powoli się goi w miarę odnowy ⟨x⟩.
- To czyni *niewidzialne* homeostatyczne sprzężenie zwrotne — utrzymujące σ ≈ 0,98 — *widzialnym*.
- *Wizualizacja:* zestawienie trybu impulsów i trybu mapy cieplnej w tej samej chwili.

### Slajd 9 — Walidacja naukowa: bez artefaktów dopasowania
- Dokładna dyskretna estymacja **Clauset–Shalizi–Newman** (MLE), x_min wybierane testem KS, bootstrapowy test zgodności + przedziały ufności.
- **Estymator rozgałęzień metodą wieloetapowej regresji (MR)** — odzyskuje prawdziwy współczynnik z dokładnością ±0,005 przy 10% podpróbkowaniu, gdzie estymator naiwny wskazuje 0,5.
- Bramka werdyktu to *klasyfikator*: przepuszcza autentyczne SOqC i **odrzuca wszystkie 7 kontroli negatywnych** (krytyczny, wygaszony, rozbieżny, dryfujące σ…).
- *Wizualizacja:* raport walidacji + wykres σ_MR(N).

### Slajd 10 — Wyniki
- **σ_MR ≈ 0,98**, stabilne i **niezmiennicze względem N** (rozstęp < 0,003 dla N = 500–2000) → samoorganizacja, a nie przypadek.
- Prawo lawin: **obcięte prawo potęgowe, τ ≈ 2,2** — odtworzone na konektomie przestrzennym.
- Uczciwy werdykt: **kwazi-krytyczny**, a nie krytyczny w sensie pola średniego. Skończone odcięcie i stromy wykładnik to sygnatura SOqC.
- *Wizualizacja:* σ_MR płaskie względem N; rozkład lawin z odcięciem; testy pola średniego wypadające poniżej zaliczenia SOqC.

### Slajd 11 — Dlaczego to ważne
- Most między **inżynierią wysokiej wydajności** a **neuronauką układów złożonych** — jeden, wielokrotnego użytku, falsyfikowalny potok.
- Zamienia abstrakcyjny mechanizm (homeostatyczną samoorganizację) w coś, co można *zobaczyć* i *przetestować* na żywo.
- Wzorzec uczciwej nauki obliczeniowej: pozwól bramce zawieść, a potem zgłoś to, co jest naprawdę prawdą.
- *Wizualizacja:* jednozdaniowe podsumowanie + adres demo na żywo.

### Slajd 12 — Co dalej / pytania
- Badania adaptacji na poziomie pojedynczego neuronu, większe N, kolaps skalowania skończonych rozmiarów, alternatywne reguły homeostazy.
- Pytanie otwarte do sali: jak daleko poniżej σ = 1 faktycznie siedzi biologia — i czy to ma znaczenie?
- *Wizualizacja:* zaproszenie do demo na żywo; kontakt / repozytorium.

---
---

# PART 2 — LINKEDIN ARTICLE

## 🇬🇧 English version

### I tried to prove the brain is critical. The data made me more honest.

There's a beautiful idea in neuroscience called the **critical-brain hypothesis**: that the cortex
computes best when it's poised on the knife's edge between order and chaos — a phase transition,
where cascades of activity ("neuronal avalanches") come in every size, with no characteristic
scale. Sit exactly on that edge and information travels furthest, dynamic range is widest, the
system is maximally responsive.

I wanted to *see* it. Not a static plot in a paper — a living network, on my screen, organizing
itself toward that edge in real time. So I built **CriticalCortex**.

**How I built it.** The simulation kernel is written in **Rust** — a zero-allocation hot loop that
pushes ~137,000 spikes per second on Apple Silicon. Around it sits a lightweight **Python** server,
and between the backend and the browser I wrote a **custom binary WebSocket protocol**: each frame
is a tightly packed little-endian payload — spike ids as `uint16`, and every neuron's resource
level as a single `uint8`. The **WebGL** frontend renders a few thousand neurons as a glowing 3D
sphere at 30 FPS, and — crucially — the simulation tick is **decoupled** from the render tick, so
the kernel runs flat-out while the browser sips a couple of kilobytes a second. When I later added a
spatially-embedded connectome (connectivity that decays with distance, conduction delays that scale
with it), the kernel didn't change a single line — it only ever reads the connectivity as flat CSR
arrays, so new topologies just drop in.

**The insight I didn't expect to be so beautiful.** The mechanism that keeps the network near the
edge is **homeostasis** — short-term synaptic depression. Every neuron carries a resource variable,
⟨x⟩, that drains when it fires and recovers slowly. It's negative feedback: burst too hard and you
deplete yourself, which quiets you down. On a spike raster this feedback is invisible. So I gave the
frontend a second rendering mode — an **⟨x⟩ heatmap**. Toggle it, and every neuron is colored by how
depleted it is: dark purple for exhausted, bright yellow for recovered. Suddenly you can *watch* an
avalanche tear across the sphere and leave a **dark "scar" of depletion behind it** — a scar that
slowly heals as resources recover. That healing *is* the self-organization. It's the invisible
machinery of criticality, made visible.

**Where the honesty comes in.** I set out to demonstrate textbook self-organized **criticality** —
the mean-field prediction of an avalanche exponent τ = 3/2 and a branching ratio σ = 1. The data
refused to cooperate. What the network actually does is settle at a branching ratio of **σ ≈ 0.98**
— stable, reproducible, and *the same across system sizes* — with a **truncated** power law and a
steeper exponent, **τ ≈ 2.2**. That's not a failed experiment. It's a different, well-documented
regime: **self-organized *quasi*-criticality (SOqC)** — the generic result when homeostatic feedback
parks a system just *below* the critical point rather than exactly on it.

I could have cropped a plot and called it 3/2. Instead I built the validation to make that
impossible. Exact discrete power-law fitting (Clauset–Shalizi–Newman), Kolmogorov–Smirnov
goodness-of-fit with bootstrap confidence intervals, likelihood-ratio tests against a truncated
alternative, and — because recording only a fraction of neurons badly biases the naive branching
estimator — a **subsampling-corrected multistep-regression estimator** that recovers the true ratio
to ±0.005 where the naive one reads 0.5. And I made the pass/fail gate a *classifier*: I fed it seven
non-SOqC surrogates (perfectly critical, dead, runaway, drifting) and confirmed it **rejects every
one**. A gate that green-lights everything proves nothing.

The most satisfying moment of the whole build wasn't the render. It was watching the mean-field
checks fail — τ ≠ 3/2, pure power law rejected — while the SOqC gate passed, and knowing the verdict
was *earned*.

**So here's what I keep coming back to:** the interesting science often lives in the gap between the
clean theory you hoped for and the messier thing the system actually does. Quasi-criticality isn't a
consolation prize — it may be exactly where biological networks sit, and *why*.

If you work in computational neuroscience, complex systems, or high-performance simulation, I'd love
your take: **how far below σ = 1 does real cortex operate — and is that sub-criticality a bug, or a
feature?** What would you want to see rendered next?

*#ComputationalNeuroscience #ComplexSystems #Rust #WebGL #Criticality #SOqC #ScientificComputing*

---

## 🇵🇱 Wersja polska

### Chciałem udowodnić, że mózg jest krytyczny. Dane nauczyły mnie większej uczciwości.

W neuronauce istnieje piękna idea — **hipoteza krytycznego mózgu**: że kora oblicza najlepiej, gdy
balansuje na ostrzu noża między porządkiem a chaosem — w przejściu fazowym, gdzie kaskady aktywności
(„lawiny neuronalne”) przyjmują każdą wielkość, bez charakterystycznej skali. Usiądź dokładnie na tej
krawędzi, a informacja niesie się najdalej, zakres dynamiczny jest najszerszy, a układ maksymalnie
reaguje.

Chciałem to *zobaczyć*. Nie statyczny wykres w artykule — żywą sieć, na moim ekranie, organizującą
się w kierunku tej krawędzi w czasie rzeczywistym. Więc zbudowałem **CriticalCortex**.

**Jak to zbudowałem.** Jądro symulacji napisałem w **Rust** — to pętla o zerowej alokacji, która na
Apple Silicon przetwarza ~137 000 impulsów na sekundę. Wokół niej działa lekki serwer w **Pythonie**,
a między backendem a przeglądarką napisałem **autorski binarny protokół WebSocket**: każda ramka to
ciasno upakowany ładunek little-endian — identyfikatory impulsów jako `uint16` oraz poziom zasobu
każdego neuronu jako pojedynczy `uint8`. Front-end **WebGL** renderuje kilka tysięcy neuronów jako
świecącą sferę 3D w 30 FPS, a — co kluczowe — takt symulacji jest **rozprzężony** od taktu
renderowania, więc jądro pracuje na pełnych obrotach, podczas gdy przeglądarka pobiera zaledwie
kilka kilobajtów na sekundę. Gdy później dodałem przestrzenny konektom (łączność malejąca z
odległością, opóźnienia przewodzenia rosnące wraz z nią), jądro nie zmieniło się o ani jedną linię —
czyta łączność wyłącznie jako płaskie tablice CSR, więc nowe topologie po prostu się wpinają.

**Wgląd, po którym nie spodziewałem się aż takiego piękna.** Mechanizmem, który utrzymuje sieć blisko
krawędzi, jest **homeostaza** — krótkoterminowa depresja synaptyczna. Każdy neuron ma zmienną zasobu
⟨x⟩, która opada, gdy neuron się wyładowuje, i wolno się odnawia. To ujemne sprzężenie zwrotne:
wybuchnij zbyt mocno, a wyczerpiesz się, co cię wycisza. Na rasterze impulsów to sprzężenie jest
niewidzialne. Dodałem więc drugi tryb renderowania — **mapę cieplną ⟨x⟩**. Przełącz ją, a każdy neuron
zostaje pokolorowany według stopnia wyczerpania: ciemny fiolet dla wyczerpanych, jasna żółć dla
odnowionych. Nagle można *oglądać*, jak lawina przetacza się przez sferę i zostawia za sobą **ciemną
„bliznę” wyczerpania** — bliznę, która powoli się goi w miarę odnowy zasobów. To gojenie *jest*
samoorganizacją. To niewidzialna maszyneria krytyczności — uczyniona widzialną.

**Tu wkracza uczciwość.** Zamierzałem zademonstrować podręcznikową samoorganizującą się
**krytyczność** — przewidywanie pola średniego: wykładnik lawin τ = 3/2 i współczynnik rozgałęzień
σ = 1. Dane odmówiły współpracy. To, co sieć faktycznie robi, to ustabilizowanie się przy
współczynniku rozgałęzień **σ ≈ 0,98** — stabilnym, powtarzalnym i *takim samym niezależnie od
rozmiaru układu* — z **obciętym** prawem potęgowym i stromszym wykładnikiem, **τ ≈ 2,2**. To nie jest
nieudany eksperyment. To odmienny, dobrze udokumentowany reżim: **samoorganizująca się
*kwazi*-krytyczność (SOqC)** — typowy wynik, gdy homeostatyczne sprzężenie zwrotne ustawia układ tuż
*poniżej* punktu krytycznego, a nie dokładnie na nim.

Mogłem przyciąć wykres i nazwać to 3/2. Zamiast tego zbudowałem walidację tak, by to uniemożliwić.
Dokładne dyskretne dopasowanie prawa potęgowego (Clauset–Shalizi–Newman), test zgodności
Kołmogorowa–Smirnowa z bootstrapowymi przedziałami ufności, testy ilorazu wiarygodności wobec
alternatywy obciętej, oraz — ponieważ rejestrowanie tylko części neuronów mocno obciąża naiwny
estymator rozgałęzień — **estymator metodą wieloetapowej regresji, skorygowany na podpróbkowanie**,
który odzyskuje prawdziwy współczynnik z dokładnością ±0,005 tam, gdzie naiwny wskazuje 0,5. Bramkę
zaliczenia uczyniłem *klasyfikatorem*: podałem jej siedem surogatów spoza SOqC (idealnie krytyczny,
wygaszony, rozbiegany, dryfujący) i potwierdziłem, że **odrzuca każdy z nich**. Bramka, która
przepuszcza wszystko, nie dowodzi niczego.

Najbardziej satysfakcjonującym momentem całej budowy nie był render. Było nim patrzenie, jak testy
pola średniego zawodzą — τ ≠ 3/2, czyste prawo potęgowe odrzucone — podczas gdy bramka SOqC
przechodzi, i świadomość, że werdykt był *zasłużony*.

**Dlatego wciąż wracam do jednej myśli:** ciekawa nauka często mieszka w szczelinie między czystą
teorią, na którą liczyłeś, a bardziej pokręconą rzeczą, którą układ faktycznie robi. Kwazi-krytyczność
nie jest nagrodą pocieszenia — może być właśnie tym, gdzie siedzą biologiczne sieci, i *dlaczego*.

Jeśli zajmujesz się neuronauką obliczeniową, układami złożonymi lub symulacjami wysokiej wydajności,
chętnie poznam Twoje zdanie: **jak daleko poniżej σ = 1 działa prawdziwa kora — i czy ta
podkrytyczność to błąd, czy cecha?** Co chciałbyś zobaczyć wyrenderowane jako następne?

*#NeuronaukaObliczeniowa #UkładyZłożone #Rust #WebGL #Krytyczność #SOqC #ObliczeniaNaukowe*
