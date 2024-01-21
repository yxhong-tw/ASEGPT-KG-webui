import { Component } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faAngleUp } from '@fortawesome/free-solid-svg-icons';

interface PageTopButtonProps {}

interface PageTopButtonState {
    showPageTop: boolean;
}

export default class PageTopButton extends Component<
    PageTopButtonProps,
    PageTopButtonState
> {
    constructor(props: PageTopButtonProps) {
        super(props);
        this.state = { showPageTop: false };
    }

    _scrollToPageTop(e: React.MouseEvent<HTMLAnchorElement>) {
        e.preventDefault();

        const target = e.target as HTMLAnchorElement;
        const targetParent = target.parentNode as HTMLAnchorElement;
        const targetGrandParent = targetParent.parentNode as HTMLAnchorElement;
        const href = (target.getAttribute('href') ||
            targetParent.getAttribute('href') ||
            targetGrandParent.getAttribute('href')) as string;

        document.querySelector(href)?.scrollIntoView({
            behavior: 'smooth',
            block: 'start',
        });
    }

    componentDidMount() {
        document.addEventListener('scroll', () => {
            this.setState({
                showPageTop: window.scrollY >= window.innerHeight / 1.2,
            });
        });
    }

    render() {
        return (
            <a
                className={
                    'scroll-to-top rounded hover:bg-gray-700 ' +
                    (this.state.showPageTop ? 'block' : 'hidden')
                }
                href="#page-top"
                onClick={this._scrollToPageTop}
            >
                <FontAwesomeIcon icon={faAngleUp} size="lg" />
            </a>
        );
    }
}
