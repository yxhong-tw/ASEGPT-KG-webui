import { useRouter } from 'next/navigation';
import Layout from '@/components/Layout';


export default function Home() {
    const router = useRouter();

    const _search = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        const inputElement = document.getElementById('base-input') as HTMLInputElement;
        const query = inputElement.value;

        if (query.trim()) {
            router.push(`/kg?query=${query}`);
        }
    };

    return (
        <Layout selectedIdx={-1}>
            <div className="flex flex-wrap p-7 mx-12">
                <form className="mb-5 inline-block w-11/12" onSubmit={_search}>
                    <label htmlFor="base-input" className="block my-3 text-lg font-medium text-gray-900 dark:text-white">請輸入欲查詢問題：</label>
                    <input type="text" id="base-input" className=" bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 inline-block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" />
                    <button type="submit" className="my-5 text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg w-full sm:w-auto px-5 py-2.5 text-center dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800">Query</button>
                </form>
            </div>
        </Layout>
    );
}
