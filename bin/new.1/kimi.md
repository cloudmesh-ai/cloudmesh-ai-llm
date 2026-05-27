
# UVA GENAI "BRIDGE" MANUAL (LOCAL-TO-REMOTE)

1. ESTABLISH THE TUNNEL (Must be active for requests to work)
   
   Run this in a dedicated terminal:
   ```bash
   ssh -L 8080:open-webui.rc.virginia.edu:443 uva
   ```

   * TIP: Add to ~/.ssh/config to use: ssh -N uva-genai

2. VS CODE / CLINE CONFIGURATION
   
   Provider: OpenAI Compatible
   Base URL: http://localhost:8080/api
   API Key:  [YOUR_KEY_FROM_~/gemma/uva-kimmi-key.txt]
   Model ID: Kimi K2.5

3. CLI AUTOMATION (Add this to your .bashrc or .zshrc)
   ```bash
   ask_kimi() {
       curl -ks -X POST "https://localhost:8080/api/chat/completions" \
            -H "Authorization: Bearer $(tr -d '[:space:]' < ~/gemma/uva-kimmi-key.txt)" \
            -H "Content-Type: application/json" \
            -H "Host: open-webui.rc.virginia.edu" \
            -d "{\"model\": \"Kimi K2.5\", \"messages\": [{\"role\": \"user\", \"content\": \"$1\"}]}"
   }
   ```

4. TROUBLESHOOTING QUICK-CHECK
   
   - 401 UNAUTHORIZED -> Check key formatting (tr -d '[:space:]')
   - CONNECTION REFUSED -> The SSH tunnel terminal is likely closed
   - SSL ERROR          -> Ensure '-k' flag is used with curl
   - 404 NOT FOUND      -> Ensure Base URL is http://localhost:8080/api

