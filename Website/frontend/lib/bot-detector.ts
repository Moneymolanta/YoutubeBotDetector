interface BotAnalysis {
  isBot: boolean
  confidence: number
  reason?: string
}

export function detectBotPattern(commentText: string): BotAnalysis {
  // Remove HTML tags for analysis
  const plainText = commentText.replace(/<[^>]*>/g, "")

  // Initialize score (0-100, higher means more likely to be a bot)
  let botScore = 0
  let reason = ""

  // Check for very short comments
  if (plainText.length < 5) {
    botScore += 30
    reason = "Very short comment"
  }

  // Check for excessive emojis or special characters
  const emojiRegex =
    /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F700}-\u{1F77F}\u{1F780}-\u{1F7FF}\u{1F800}-\u{1F8FF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu
  const emojiCount = (plainText.match(emojiRegex) || []).length
  const textLength = plainText.length

  if (emojiCount > 0 && emojiCount / textLength > 0.3) {
    botScore += 25
    reason = reason ? `${reason}, Excessive emojis` : "Excessive emojis"
  }

  // Check for suspicious links or spam patterns
  const spamPatterns = [
    /check out my channel/i,
    /subscribe to my/i,
    /follow me on/i,
    /check my profile/i,
    /make money/i,
    /earn \$\d+/i,
    /click here/i,
    /bit\.ly/i,
    /goo\.gl/i,
    /t\.co/i,
    /cutt\.ly/i,
    /tinyurl/i,
  ]

  for (const pattern of spamPatterns) {
    if (pattern.test(plainText)) {
      botScore += 35
      reason = reason ? `${reason}, Contains spam patterns` : "Contains spam patterns"
      break
    }
  }

  return {
    isBot: botScore >= 50,
    confidence: botScore,
    reason: reason || "No specific patterns detected",
  }
}
