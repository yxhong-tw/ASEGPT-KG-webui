import { Component } from 'react';
import Link from 'next/link';
import Image from 'next/image';

interface NavbarProps {
    selectedIdx?: number;
}

interface NavbarState {
    mode: 'light' | 'dark';
}

export default class Navbar extends Component<NavbarProps, NavbarState> {
    constructor(props: NavbarProps) {
        super(props);

        this.state = {
            mode: 'light',
        } as NavbarState;
    }

    _activateSelected(target: number = 0) {
        return this.props.selectedIdx === target
            ? 'bg-white text-black font-semibold'
            : 'text-white hover:text-gray-200';
    }

    render() {
        const basicLinkStyle =
            ' px-3 py-2 rounded-md font-medium font-sans tracking-wide ';

        const { mode } = this.state;

        return (
            <div className="navbar w-full z-60 shadow">
                <nav className="h-full">
                    <div className="h-full mx-auto py-1 px-2 sm:px-6 lg:px-12 lg:pr-36">
                        <div className="h-full relative flex flex-col xl:flex-row items-center justify-between">
                            <div className="h-full flex-1 flex items-center justify-center sm:items-stretch sm:justify-start">
                                <div className="flex-shrink-0 flex items-center relative">
                                    <Link
                                        href="/"
                                        className="p-1 rounded-xl flex"
                                    >
                                        {/* <div className="relative top-1">
                                            <Image
                                                src={'/ASE_Technology_logo.png'}
                                                alt="ASE icon"
                                                width={70}
                                                height={70}
                                            />
                                        </div> */}
                                        <div className="p-5 pl-7 mt-1 text-white font-sans tracking-wide text-3xl font-semibold cursor-pointer">
                                            ASEGPT-KG
                                        </div>
                                    </Link>
                                </div>
                            </div>
                            <div className="hidden sm:block sm:ml-6 relative tracking-wide text-center">
                                <div className="flex space-x-4 last:ml-20 gap-3">
                                    <Link
                                        href="/kg"
                                        className={
                                            this._activateSelected(0) +
                                            basicLinkStyle
                                        }
                                    >
                                        Graph
                                    </Link>
                                    <Link
                                        href="/gpt"
                                        className={
                                            this._activateSelected(1) +
                                            basicLinkStyle
                                        }
                                    >
                                        GPT
                                    </Link>
                                    <button
                                        id="dark"
                                        className={
                                            'px-4 py-2 rounded-full border-2 ' +
                                            (mode === 'dark'
                                                ? ' bg-gray-800 border-gray-700 text-white hover:bg-white hover:text-black'
                                                : ' bg-white border-gray-400 text-black hover:bg-gray-700 hover:text-white')
                                        }
                                        onClick={() => {
                                            document.documentElement.classList.toggle(
                                                'dark'
                                            );
                                            document
                                                .getElementsByTagName('nav')[0]
                                                .classList.toggle('dark');

                                            this.setState({
                                                mode:
                                                    mode === 'light'
                                                        ? 'dark'
                                                        : 'light',
                                            });
                                        }}
                                    >
                                        {mode === 'dark' ? '夜間' : '日間'}
                                        模式
                                        <i className="fas fa-moon text-yellow-500"></i>
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </nav>
            </div>
        );
    }
}
