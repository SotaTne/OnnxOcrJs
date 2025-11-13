declare module "opencv-react" {
  interface OpenCvInstance {
    loaded: boolean;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    cv: any;
  }

  interface OpenCvProviderProps {
    openCvVersion?: string;
    openCvPath?: string;
    children: React.ReactNode;
  }

  export const OpenCvContext: React.Context<OpenCvInstance>;

  export const OpenCvConsumer: React.Consumer<OpenCvInstance>;

  export const OpenCvProvider: React.FC<OpenCvProviderProps>;

  export function useOpenCv(): OpenCvInstance;
}
