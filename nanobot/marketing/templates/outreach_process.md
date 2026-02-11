Sehr geehrte/r {{ anrede|default('Frau/Herr') }} {{ nachname|default('') }},

die Optimierung von Geschaeftsprozessen ist ein zentraler Hebel fuer nachhaltige Wettbewerbsfaehigkeit{% if signal_description %} -- ein Thema, das offenbar auch fuer {{ company_name }} relevant ist: {{ signal_description }}{% endif %}.

Bei Kraus & Partner verbinden wir Prozessexzellenz mit Veraenderungskompetenz. Denn die besten Prozesse nutzen wenig, wenn die Organisation sie nicht lebt. Wir helfen Ihnen, beides zu erreichen.

{% if consultant_name %}{{ consultant_name }} hat zahlreiche Prozessoptimierungsprojekte{% if industry %} in der {{ industry }}{% endif %} erfolgreich begleitet und steht fuer ein Gespraech bereit.{% endif %}

Duerfen wir einen Termin vereinbaren?

Mit freundlichen Gruessen
{{ absender_name }}
Kraus & Partner

---
{{ impressum }}

Sie moechten keine weiteren Nachrichten erhalten? {{ abmeldelink }}
