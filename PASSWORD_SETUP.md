# Password Protection Setup for EMI Shielder

## Local Development

1. The app is now password protected. The default password is in `.streamlit/secrets.toml`
2. To change the password locally, edit `.streamlit/secrets.toml`
3. The secrets file is gitignored and will NOT be uploaded to GitHub

## Deployment on Free Hosting Services

### Option 1: Streamlit Community Cloud (Recommended)
1. Deploy to [share.streamlit.io](https://share.streamlit.io)
2. In your app settings, go to "Secrets" section
3. Add your secrets in TOML format:
   ```toml
   password = "YourSecurePassword123!"
   ```
4. The app will use these secrets automatically

### Option 2: Hugging Face Spaces
1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Upload your code
3. Go to Settings → Repository secrets
4. Add `STREAMLIT_SECRETS_PASSWORD` with your password value

### Option 3: Render.com
1. Create account at [render.com](https://render.com)
2. Connect GitHub repo
3. Add environment variable:
   - Key: `PASSWORD`
   - Value: Your secure password

### Option 4: Railway.app
1. Sign up at [railway.app](https://railway.app)
2. Deploy from GitHub
3. Add environment variables in the dashboard

## Security Best Practices

1. **Use strong passwords**: At least 12 characters with mix of letters, numbers, symbols
2. **Change default password**: Never use the example password in production
3. **Use environment variables**: For cloud deployments, always use platform-specific secrets management
4. **Enable HTTPS**: Most platforms provide HTTPS by default
5. **Consider timeout**: The auth system can auto-logout after 30 minutes of inactivity

## Advanced Features

### Session Timeout
To enable 30-minute session timeout, change in app.py:
```python
from auth import check_password_with_timeout

# Replace check_password() with:
if not check_password_with_timeout(timeout_minutes=30):
    st.stop()
```

### Multi-User Access
For different access levels, use:
```python
from auth import check_multi_user_password

if not check_multi_user_password():
    st.stop()

# Then check access level:
if st.session_state.get("access_level") == "readonly":
    # Show limited features
```

## Environment Variables Alternative

Instead of secrets.toml, you can use environment variables:

```python
import os

# In your auth check:
correct_password = os.environ.get("EMI_PASSWORD", "default_password")
```

Then set the environment variable when deploying.

## Testing Password Protection

1. Run the app: `streamlit run streamlit_app/app.py`
2. You'll see a password prompt
3. Enter the password from `.streamlit/secrets.toml`
4. The app will load after successful authentication

## Important Notes

- Never commit passwords to git
- Always use HTTPS in production
- Consider using OAuth for enterprise deployments
- Rotate passwords regularly
- Monitor access logs if available on your hosting platform