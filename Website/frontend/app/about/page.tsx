import { Bot, Database, Brain } from "lucide-react"

export default function AboutPage() {
  return (
    <div className="max-w-5xl mx-auto p-6 space-y-8">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold">About Our Bot Detection Models</h1>
        <p className="text-lg text-gray-600 max-w-3xl mx-auto">
          Our platform uses advanced AI models to analyze YouTube comments and identify potential bot activity
        </p>
      </div>

      <div className="bg-white rounded-lg shadow-sm p-6">
        <div className="prose max-w-none">
          <p>
            Our models are trained on a dataset of comments from Kaggle:
            <a
              href="https://www.kaggle.com/datasets/example/youtube-comments"
              className="text-blue-600 hover:underline ml-1"
              target="_blank"
              rel="noopener noreferrer"
            >
              link to dataset
            </a>
            .
          </p>

          <p>
            The dataset contains comments from popular music videos with labels on whether the comments were made by
            bots or humans.
          </p>

          {/* <h2 className="text-xl font-semibold mt-6 mb-3">Dataset Parameters:</h2>
          <ul className="list-disc pl-6 space-y-1">
            <li>Comment ID</li>
            <li>Commenter Username</li>
            <li>Comment Text</li>
            <li>Date</li>
            <li>Video Name</li>
            <li>Label (Used only for validation)</li>
          </ul> */}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="h-12 w-12 rounded-full bg-blue-100 flex items-center justify-center mb-4">
            <Bot className="h-6 w-6 text-blue-600" />
          </div>
          <h3 className="text-lg font-semibold mb-2">CNN Model</h3>
          <p className="text-gray-600">
            Our Convolutional Neural Network (CNN) model analyzes patterns in comment text to identify bot-like
            behavior. It's particularly effective at detecting repetitive patterns and unusual character usage.
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="h-12 w-12 rounded-full bg-purple-100 flex items-center justify-center mb-4">
            <Database className="h-6 w-6 text-purple-600" />
          </div>
          <h3 className="text-lg font-semibold mb-2">RNN Model</h3>
          <p className="text-gray-600">
            The Recurrent Neural Network (RNN) model examines the sequential nature of text to identify unnatural
            language patterns that may indicate automated content generation.
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="h-12 w-12 rounded-full bg-green-100 flex items-center justify-center mb-4">
            <Brain className="h-6 w-6 text-green-600" />
          </div>
          <h3 className="text-lg font-semibold mb-2">BERT Model</h3>
          <p className="text-gray-600">
            Our BERT (Bidirectional Encoder Representations from Transformers) model leverages contextual understanding
            of language to detect subtle indicators of bot-generated content that simpler models might miss.
          </p>
        </div>
      </div>

      {/* <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-xl font-semibold mb-4">Flask Integration</h2>
        <p className="mb-4">
          Our platform is designed to integrate with a Flask backend for more advanced AI processing. The Flask backend
          can:
        </p>
        <ul className="list-disc pl-6 space-y-1">
          <li>Run more complex machine learning models</li>
          <li>Process larger datasets more efficiently</li>
          <li>Provide more detailed analytics and insights</li>
          <li>Scale to handle higher traffic volumes</li>
        </ul>

        <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
          <h3 className="text-lg font-medium mb-2">API Integration</h3>
          <p>
            The Next.js frontend communicates with the Flask backend through RESTful API endpoints, allowing for
            seamless data exchange and processing.
          </p>
        </div>
      </div> */}
    </div>
  )
}
