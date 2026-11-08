# Domain deployment: `kevexploit.us`

## Current status

The domain `kevexploit.us` is already registered on Porkbun and is owned by Luis's account.

Luis will keep ownership of the registrar account. Do not request or share:

- Porkbun password
- 2FA codes
- Full registrar access

## Chen's task

Deploy the website using the hosting provider of choice. Recommended options:

- GitHub Pages for the static dashboard
- Cloudflare Pages, Netlify, or Vercel for static hosting
- A separate backend host only if the repository scanner needs live server-side execution

After deployment, send Luis:

1. Hosting provider
2. Production URL
3. Exact DNS records required
4. Whether `www.kevexploit.us` should redirect to the root domain
5. Any required verification TXT record

Do not guess DNS values. Send the exact records provided by the hosting provider.

## Luis's task after Chen sends the records

Luis will open Porkbun DNS for `kevexploit.us` and add the records exactly as provided.

Typical examples are:

### GitHub Pages

```text
A      @      185.199.108.153
A      @      185.199.109.153
A      @      185.199.110.153
A      @      185.199.111.153
CNAME  www    <github-user>.github.io
```

Chen must also configure the custom domain in GitHub:

```text
Repository Settings -> Pages -> Custom domain -> kevexploit.us
```

### Other hosting providers

The provider may instead request a CNAME, A record, or TXT verification record. Use the provider's instructions exactly.

## Security checklist

- Enable Porkbun 2FA.
- Enable domain lock.
- Keep auto-renew enabled if the team wants the domain to remain active.
- Never publish registrar credentials or personal contact information.
- Do not enable URL forwarding unless explicitly needed.
- Do not change nameservers unless the hosting provider requires it.

## Final verification

After DNS propagation:

1. Open `https://kevexploit.us`.
2. Open `https://www.kevexploit.us`.
3. Confirm HTTPS is enabled.
4. Confirm the dashboard loads its data files.
5. Confirm the repository scanner is either working or clearly labeled as a separate backend demo.

## Message to send Chen

> I registered `kevexploit.us` on Porkbun. I will keep ownership of the domain and will not share registrar credentials. Please deploy the site and send me the exact DNS records from your hosting provider, plus any TXT verification record and the intended `www` redirect. I will add those records in Porkbun and enable HTTPS after DNS propagation.
