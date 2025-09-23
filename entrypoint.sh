#!/bin/bash
set -e

# Get UID/GID from environment variables, with a default of 1000
USER_ID=${UID:-1000}
GROUP_ID=${GID:-1000}

# Check if the group exists, if not create it
if ! getent group appgroup >/dev/null; then
    groupadd -g $GROUP_ID appgroup
fi

# Check if the user exists, if not create it
if ! id -u appuser >/dev/null 2>&1; then
    useradd --shell /bin/bash -u $USER_ID -g $GROUP_ID -o -c "" -m appuser
else
    # User exists, modify UID/GID to match the host
    groupmod -g $GROUP_ID appgroup
    usermod -u $USER_ID -g $GROUP_ID appuser
fi

# Fix permissions on home and app directories
# This ensures the container user can write to the mounted volumes
chown -R $USER_ID:$GROUP_ID /home/appuser
chown -R $USER_ID:$GROUP_ID /app

# Execute the command passed to the script (e.g., streamlit run...)
exec gosu appuser "$@"
