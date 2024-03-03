import Head from 'next/head';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import PageTopButton from '@/components/Pagetop';
import LoadingSpinner from '@/components/LoadingSpinner';
import Blocker from '@/components/Blocker';

interface LayoutProps {
    children: React.ReactNode;
    title?: string;
    openPanel?: boolean;
    selectedIdx?: number;
}

export default function Layout(props: LayoutProps) {
    const {
        children,
        title = 'ASEGPT-KG',
        openPanel = null,
        selectedIdx = undefined,
    } = props;

    return (
        <div>
            <Head>
                <title>{title}</title>
                <meta name="description" content="ASEGPT-KG" />
                <meta
                    name="viewport"
                    content="width=device-width, initial-scale=1"
                />
                <meta charSet="utf-8" />
                <meta
                    name="viewport"
                    content="initial-scale=1.0, width=device-width"
                />
            </Head>
            <div className="w-full h-screen flex flex-wrap bg-blue-200/20">
                <Navbar selectedIdx={selectedIdx} />
                <main id="page-top" className="mainContainer w-full">
                    {children}
                    {openPanel === null ? null : (
                        <Blocker
                            display={openPanel}
                            style={{ backgroundColor: 'rgba(0, 0, 0, .2)' }}
                        >
                            <LoadingSpinner />
                        </Blocker>
                    )}
                </main>
                <Footer />
            </div>
            <PageTopButton />
        </div>
    );
}
