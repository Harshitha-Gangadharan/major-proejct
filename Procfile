web: gunicorn Cloud.wsgi:application --bind 0.0.0.0:$PORT

# Explicit WSGI module to avoid an empty `WSGI_MODULE` causing startup errors.

# If you change your project package name, update this line to `<yourpkg>.wsgi:application`.
