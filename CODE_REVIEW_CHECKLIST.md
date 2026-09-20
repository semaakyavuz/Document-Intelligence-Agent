# Kod İnceleme Kontrol Listesi

Bu projede bir değişiklik tamamlandığında, birleştirmeden önce aşağıdaki altı başlık
tek tek gözden geçirilir. Amaç eksiksizlik değil, bu projede **gerçekten sorun çıkmış**
noktaları yakalamak; her maddenin altındaki örnekler yaşanmış vakalardan geliyor.

---

## 1. Hata yönetimi

- [ ] Her `fetch` / ağ çağrısı `try/catch` içinde mi ve `response.ok` kontrol ediliyor mu?
      (Sunucuya hiç ulaşılamaması ile sunucunun hata döndürmesi ayrı durumlar, ayrı mesaj isterler.)
- [ ] Akış (SSE) ayrıştırıcıları bozuk/yarım bir satırda tüm akışı düşürmek yerine o satırı atlıyor mu?
- [ ] Bir işlemi başlatırken devre dışı bırakılan düğmeler `finally` ile tekrar etkinleştiriliyor mu?
- [ ] Listeyi render eden fonksiyonlar boş/eksik/`null` veride çökmeden sessizce gizleniyor mu?
- [ ] **Olmayan durumlar için savunma yazılmıyor.** Sınır (kullanıcı girdisi, dış API) dışında
      iç koda güvenilir; "olur da" diye yazılan kontroller ölü koda dönüşür.

## 2. Güvenlik

- [ ] **`innerHTML`'e giren her değer kaynağına kadar takip edildi mi?**
      Kaçışsız tek bir değer yeter. Bu projede iki kez yaşandı:
  - `file.name` (kullanıcı dosya adı) → toplu yükleme satırlarında,
  - `seller_name` / `invoice_no` (**fatura görselinden çıkarılıyor**, yani içeriği dışarıdan
    belirlenebilir) → panel tablosunda kalıcı (stored) XSS.
- [ ] Dinamik veri `createElement` + `textContent` ile mi yazılıyor?
      `innerHTML` yalnızca kod içinde yazılmış sabit dizeler için kullanılır.
- [ ] Yeni bir dış kaynak (CDN, font, API) ekleniyorsa: gerekli mi, ve **tam sürüme sabit mi**?
      Kayan sürüm (`@4` gibi) sayfayı habersiz değiştirebilir.
- [ ] Gizli bilgi (anahtar, token) koda ya da commit'e sızmıyor mu? `.env` dışına çıkmamalı.

## 3. Performans varsayımı

- [ ] "Küçük kalır" varsayımı yazılı mı ve doğru mu? (Örn. toplu yükleme en fazla 10 dosya,
      panel `limit=100` — bu sınırlar kalkarsa render stratejisi de değişmeli.)
- [ ] Her tuş vuruşunda / her olayda tüm listeyi yeniden çizen bir yol var mı? Sınırı biliniyor mu?
- [ ] Tekrar çağrıldığında eski nesneyi temizlemeyen bir kurulum var mı?
      (Örn. aynı canvas üzerine ikinci bir `new Chart(...)`.)
- [ ] Dış font/betik için `preconnect` verilmiş mi?
- [ ] Ölçmeden "yavaş/hızlı" denmiyor; iddia varsa ölçüm de var.

## 4. Yeni bağımlılık gerekliliği

- [ ] Gerçekten gerekli mi, yoksa birkaç satır kendi kodumuzla çözülür mü?
- [ ] Frontend **vanilla JS + CSS + Chart.js** ile sınırlı; React/Vue/Next.js gibi bir
      çatı eklenmiyor, derleme adımı gerektirmiyor.
- [ ] Python tarafında yeni paket `requirements.txt`'e işlendi mi ve projenin kendi
      `.venv`'i ile kurulup doğrulandı mı?
- [ ] Yalnızca geliştirme/doğrulama için kullanılan araçlar (örn. tarayıcı otomasyonu)
      proje bağımlılığı **yapılmadı**, geçici dizinde tutuldu mu?

## 5. Mimari uyum

- [ ] **Ajanlar, sunucular ve graph kurucular `Settings`'i içeriden okumaz.** Fabrika
      fonksiyonları bağımlılıkları parametre olarak alır, varsayılanı gerçek olandır:
      `build_server(settings: Settings | None = None, embedding_provider: ... = None)`
- [ ] Frontend, backend'in gerçekten gönderdiği alanları mı tüketiyor?
      Gönderilmeyen bir ara durum ("şu an düşünüyor" gibi) uydurulmuyor.
- [ ] Backend sözleşmesi (`/invoices/stream`, `/invoices/bulk`, `final_report` alanları)
      değişiyorsa **önce açıkça onay alındı mı?** Görsel/yapısal bir iş, sessizce backend
      değiştirmenin gerekçesi değildir.
- [ ] Tüm dosya yazımları `encoding="utf-8"`; Türkçe karakter basan betikler
      `sys.stdout.reconfigure(encoding="utf-8")` çağırıyor.
      (`tests/test_encoding_hygiene.py` bunu AST üzerinden denetler.)
- [ ] Testler gerçek dış servise bağlanmıyor. İstisnalar: yerel ChromaDB, yerel MCP stdio
      alt süreci ve alt sistem başına `@pytest.mark.slow` ile işaretlenmiş testler
      (`pytest.ini`'deki `markers` bölümüne kayıtlı).

## 6. Temizlik

- [ ] Kullanılmayan CSS değişkeni / sınıfı / kuralı kaldı mı?
      **Göz kararıyla değil, tarayarak bak:** tanımlı seçicileri HTML + JS kullanımına
      karşı karşılaştır. Dikkat — JS'te şablon dizesiyle üretilen sınıflar
      (`` `risk-badge--${level}` ``) aramada "kullanılmıyor" gibi görünür, silinmemeli.
- [ ] Bir önceki tasarım/palet turundan artakalan değer var mı? (Eski renk/font tanımları.)
- [ ] Tanımlanıp hiç çağrılmayan fonksiyon/sabit var mı?
- [ ] "İleride lazım olur" diye eklenmiş, bugün kullanılmayan soyutlama var mı? Silinir.
- [ ] Yorumlar **neden**'i mi anlatıyor? Ne yaptığını anlatan, kodu tekrar eden ya da
      geçmiş bir turdan kalmış yorumlar temizlendi mi?
- [ ] `console.log` / `debugger` / unutulmuş `TODO` kalmadı mı?
- [ ] Geçici dosyalar proje dizinine sızmadı mı? (`git status --untracked-files=all` ile bak.)

---

## Bitirme adımları

- [ ] `.venv/Scripts/python.exe -m pytest -m "not slow" -q` — yeşil, ve sayı beklenen
      değerde (backend'e dokunulmadıysa sayı **değişmemeli**).
- [ ] Frontend değiştiyse gerçek tarayıcıda doğrulandı mı? Varsayım değil, ölçüm:
      hesaplanan stiller, öğe sayıları, konsol hatası sıfır, dar ekranda yatay taşma yok.
- [ ] `graphify update .` çalıştırıldı mı?
- [ ] Değişen/eklenen dosyalar özetlendi mi?
