# Deploy to Render - Step by Step

## Step 1: Create GitHub Account
1. Go to https://github.com
2. Sign up (free)

## Step 2: Upload Your Project to GitHub
1. Create new repository named "banana-inspection"
2. Upload all files from your banana folder
3. Make sure to include:
   - app.py
   - requirements.txt
   - render.yaml
   - templates/ folder
   - static/ folder
   - model/best.pt

## Step 3: Deploy on Render
1. Go to https://render.com
2. Sign up with GitHub (free)
3. Click "New" → "Web Service"
4. Connect your GitHub repository
5. Render will auto-detect settings from render.yaml
6. Click "Create Web Service"
7. Wait 5-10 minutes for deployment

## Step 4: Your App is Live!
You'll get a URL like: https://banana-inspection.onrender.com

## Notes:
- Free tier: App sleeps after 15 min of inactivity
- First request after sleep takes 30-60 seconds to wake up
- Completely free forever!

## Troubleshooting:
If deployment fails, check:
- All files uploaded to GitHub
- model/best.pt file is included
- requirements.txt is correct
