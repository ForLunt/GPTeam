import React, {useEffect, useRef, useState} from "react";
import {Button, Header, Icon, Segment, Image, Modal, Input} from "semantic-ui-react";

function Picture({image}: {image: File}) {
    const [open, setOpen] = useState(false);
    const [promptOne, setPromptOne] = useState("");
    const [promptTwo, setPromptTwo] = useState("");

    function request(){
        const fdata = new FormData();

		
        fdata.append(image.name, image);
        fdata.append("prompt", promptOne);
        fdata.append("prompt2", promptTwo);
		
		fetch(`http://localhost:5000/retouche`, {
			body: fdata,
			method: "POST",
		});
    }

    return (
		<>
			<Modal onClose={() => setOpen(false)} onOpen={() => setOpen(true)} open={open} trigger={<Image src={URL.createObjectURL(image)} />}>
				<Modal.Header>Modification de la photo</Modal.Header>
				<Modal.Content image>
					<Image size="medium" src={URL.createObjectURL(image)} wrapped />
					<Modal.Description>
						<Header>Modification de l'image</Header>
						<p>Vous pouvez modifier l'image en choisisant quel élément enlever</p>
						<p>Et choisir par quoi le remplacer</p>
						<Input onChange={(e, data) => setPromptOne(data.value)} fluid type="text" label="L'élément à remplacer : " />
						<Input onChange={(e, data) => setPromptTwo(data.value)} fluid type="text" label="Par quoi le remplacer : " />
					</Modal.Description>
				</Modal.Content>
				<Modal.Actions>
					<Button color="black" onClick={() => setOpen(false)}>
						Quitter
					</Button>
					<Button content="Valider" labelPosition="right" icon="checkmark" onClick={() => {setOpen(false); console.log(promptOne + promptTwo); request()}} positive />
				</Modal.Actions>
			</Modal>
		</>
	);
}

function WrongImages({Images, fileNames}: {Images: FileList; fileNames: string[]}) {
	const [wrongImages, setWrongImages] = useState<any>();

	useEffect(() => {
		if (Images === undefined) return;
		// On récupère la liste des images qui ont pour nom un des noms de fileNames
		const wrongImages_bis = Array.from(Images).filter(image => fileNames.includes(image.name));

		setWrongImages(wrongImages_bis);
	}, [Images, fileNames]);

	return (
		<Segment inverted color="red">
			<Header inverted>
				Les images suivantes ne sont pas conservables :
			</Header>
			<Image.Group size="medium">{wrongImages !== undefined ? wrongImages.map((image: File) => <Image src={URL.createObjectURL(image)} />) : <></>}</Image.Group>
		</Segment>
	);
}

function OkImages({Images, fileNames}: {Images: FileList; fileNames: string[]}) {
	const [wrongImages, setWrongImages] = useState<any>();

	useEffect(() => {
		if (Images === undefined) return;
		// On récupère la liste des images qui ont pour nom un des noms de fileNames
		const wrongImages_bis = Array.from(Images).filter(image => fileNames.includes(image.name));

		setWrongImages(wrongImages_bis);
	}, [Images, fileNames]);

	return (
		<Segment inverted color="green">
			<Header inverted>Les images suivantes sont bonnes :</Header>
			<Image.Group size="medium">{wrongImages !== undefined ? wrongImages.map((image: File) => <Image src={URL.createObjectURL(image)} />) : <></>}</Image.Group>
		</Segment>
	);
}

function RetouchImages({Images, fileNames}: {Images: FileList; fileNames: string[]}) {
	const [wrongImages, setWrongImages] = useState<any>();

	useEffect(() => {
		if (Images === undefined) return;
		// On récupère la liste des images qui ont pour nom un des noms de fileNames
		const wrongImages_bis = Array.from(Images).filter(image => fileNames.includes(image.name));

		setWrongImages(wrongImages_bis);
	}, [Images, fileNames]);

	return (
		<Segment inverted color="yellow">
			<Header inverted>Les images suivantes doivent être retouchées :</Header>
			<Image.Group size="medium">{wrongImages !== undefined ? wrongImages.map((image: File) => <Picture image={image} />) : <></>}</Image.Group>
		</Segment>
	);
}

function Uploader() {
	const uploadRef = useRef<HTMLInputElement>(null);
	const [images, setImages] = useState<FileList | undefined>();
	const [fileNames, setFileNames] = useState<any>();
    const [loading, setLoading] = useState(false);

	useEffect(() => {
		if (images !== undefined) {
			console.log(images);
			const fdata = new FormData();

			for (let i = 0; i < images.length; i++) {
				const file = images[i];
				fdata.append(file.name, file);
			}
            setLoading(true);
			fetch(`http://localhost:5000/upload`, {
				body: fdata,
				method: "POST",
			})
				.then(res => res.json())
				.then(res => {
					console.log(res);
					setFileNames(res);
                    setLoading(false);
				});
		}
	}, [images]);
	return (
		<Segment placeholder loading={loading}>
			<Header icon>
				<Icon name="file image outline" />
				Ajouter des images
			</Header>
			<Button primary onClick={() => uploadRef.current.click()}>
				Uploader
			</Button>
			<input
				type="file"
				onChange={e => {
					if (e.target.files != undefined) setImages(e.target.files);
				}}
				ref={uploadRef}
				multiple
				style={{display: "none"}}
				accept=" image/jpeg"
			/>
			{fileNames !== undefined && (
				<>
					{fileNames.exclure.length !== 0 ? <WrongImages Images={images} fileNames={fileNames.exclure} /> : <></>}
					{fileNames.retoucher.length !== 0 ? <RetouchImages Images={images} fileNames={fileNames.retoucher} /> : <></>}
					{fileNames.ok.length !== 0 ? <OkImages Images={images} fileNames={fileNames.ok} /> : <> </>}
				</>
			)}
		</Segment>
	);
}

export default Uploader;
