css = '''
<style>
.chat-message {
    padding: 1.1rem;
    border-radius: 1.25rem;
    margin-bottom: 1rem;
    display: flex;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    font-family: 'Segoe UI', sans-serif;
}

.chat-message.user {
    background: linear-gradient(145deg, #dceefb, #e6f4ff);
    justify-content: flex-end;
}

.chat-message.bot {
    background: linear-gradient(145deg, #f2f2f2, #e0e0e0);
    justify-content: flex-start;
}

.chat-message .avatar {
    width: 60px;
    height: 60px;
    flex-shrink: 0;
    margin-right: 1rem;
}

.chat-message.user .avatar {
    order: 2;
    margin-left: 1rem;
    margin-right: 0;
}

.chat-message .avatar img {
    max-width: 60px;
    max-height: 60px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #ccc;
}

.chat-message .message {
    max-width: 75%;
    font-size: 16px;
    line-height: 1.6;
    padding: 0.5rem 1rem;
    border-radius: 1rem;
    background: white;
    color: #333;
}

.expander {
    height: 60vh;
    overflow-y: scroll;
    padding-right: 1rem;
    scroll-behavior: smooth;
}
</style>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.pinimg.com/736x/8b/16/7a/8b167af653c2399dd93b952a48740620.jpg">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

expander_css = '<style>[data-testid="stExpander"] div:has(>.streamlit-expanderContent) {overflow: auto; height: 60vh;}</style>'
