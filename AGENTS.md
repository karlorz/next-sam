# Project Overview

This is a Next.js application that performs image segmentation using Meta's Segment Anything Model V2 (SAM2) and onnxruntime-web. All the processing is done on the client side. The application allows users to upload an image, encode it, and then decode masks by clicking on the image. The application supports positive and negative point prompts for mask decoding, and it also allows users to crop the image using the decoded mask.

## Key Technologies

*   **Next.js:** A React framework for building server-side rendered and statically generated web applications.
*   **React:** A JavaScript library for building user interfaces.
*   **onnxruntime-web:** A JavaScript library for running ONNX models in the browser.
*   **SAM2:** Meta's Segment Anything Model V2 for image segmentation.
*   **Web Workers:** Used to offload heavy computations from the main thread to improve performance.
*   **Tailwind CSS:** A utility-first CSS framework for rapid UI development.
*   **shadcn/ui:** A collection of reusable UI components.

## Architecture

The application is structured as a typical Next.js project.

*   **`app/` directory:** Contains the main application logic.
    *   `page.jsx`: The main page of the application, which contains the UI and the logic for interacting with the user.
    *   `SAM2.js`: A class that encapsulates the logic for interacting with the SAM2 model. It handles model downloading, session creation, image encoding, and mask decoding.
    *   `worker.js`: A web worker that runs the SAM2 model in a separate thread to avoid blocking the main UI thread.
*   **`components/` directory:** Contains reusable UI components, such as buttons, cards, and dialogs.
*   **`lib/` directory:** Contains utility functions, such as image manipulation functions.
*   **`public/` directory:** Contains static assets, such as images.

# Building and Running

1.  **Install dependencies:**
    ```bash
    npm install
    ```
2.  **Run the development server:**
    ```bash
    npm run dev
    ```
3.  **Open your browser and visit:**
    ```
    http://localhost:3000
    ```

## Scripts

*   `npm run dev`: Starts the development server.
*   `npm run build`: Builds the application for production.
*   `npm run start`: Starts the production server.
*   `npm run lint`: Lints the code.

# Development Conventions

*   **Coding Style:** The code follows the standard JavaScript and React conventions.
*   **UI:** The UI is built using shadcn/ui components and Tailwind CSS.
*   **State Management:** The application uses React's built-in state management features (`useState`, `useEffect`, `useRef`).
*   **Web Worker:** The application uses a web worker to run the SAM2 model, which is a good practice for performance-intensive tasks.
